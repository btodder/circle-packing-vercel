"""
Flask application for serving the circle packing web app on Render or other
standard Python hosts.

This module wires together the core simulated annealing packing logic with a
simple JSON API and a static front‑end. The front‑end lives in the
``frontend`` folder and is served at the root URL (``/``). The API lives at
``/api/run`` and accepts a JSON payload describing the simulation parameters.
It returns a JSON response containing the best energy and a base64‑encoded PNG
image of the resulting packing layout.

When run directly (e.g. ``python app.py``), the application will listen on
port 10000 by default. On Render, you can configure the start command to
``python app.py`` or use a WSGI server such as Gunicorn with ``gunicorn
app:app``.
"""

import base64
import io
import threading
import time
import uuid
from flask import Flask, request, jsonify
import threading
import time
import uuid

from api.packing_core import (
    run_two_phase_optimization,
    render_packing_image,
    DEFAULT_N,
    DEFAULT_R_CONTAINER,
    DEFAULT_R_CIRCLE,
    DEFAULT_TEMP,
    DEFAULT_ITERATIONS,
    DEFAULT_NUM_STARTS,
    DEFAULT_QUICK_SCREENING_ITERATIONS,
    SOLUTION_ENERGY_THRESHOLD,
)

# Create the Flask application. ``static_folder`` points to the front‑end,
# ``static_url_path`` is set to an empty string so that ``/`` serves
# ``index.html`` from the static folder by default.
app = Flask(__name__, static_folder="frontend", static_url_path="")

# ---------------------------------------------------------------------------
# Job management for asynchronous optimization
#
# Each simulation request creates a unique job id and spawns a background
# thread to perform the optimization. Progress for phase 1 and phase 2 is
# tracked separately, along with elapsed times, energy estimates and final
# results. Clients should POST to ``/api/start`` to begin a job and then
# poll ``/api/progress/<job_id>`` to retrieve progress and, eventually, the
# completed image and success status.

# In‑memory store of jobs. Keys are job IDs (strings) and values are dicts
# holding job state. A job dict contains the following keys:
#   status: "running", "completed", or "error"
#   phase: 1 or 2
#   phase1_progress, phase2_progress: floats in [0, 1]
#   phase1_start_time, phase2_start_time: float timestamps
#   phase1_elapsed, phase2_elapsed: floats (seconds)
#   energy: current best energy (float)
#   success: bool or None
#   image: base64 string of PNG or None
#   iterations, num_starts, quick_iterations: ints (input parameters)
#   error: error message if status == "error"
jobs: dict[str, dict] = {}


def _simulate_job(job_id: str, params: dict) -> None:
    """Background thread target for running a packing optimization.

    Performs the two‑phase simulated annealing and updates the global
    ``jobs`` dict with progress and results. A ``progress_callback`` is
    registered with the core algorithm so that progress updates are sent back
    into the job entry. On completion, the final energy, success flag and
    encoded image are stored. If any exception occurs, the job status is
    marked as ``error`` and the error message is recorded.

    Parameters
    ----------
    job_id : str
        Unique identifier for the job in the ``jobs`` dict.
    params : dict
        Simulation parameters extracted from the request payload. Must
        include keys ``N``, ``R_container``, ``R_circle``, ``initial_temp``,
        ``iterations``, ``num_starts``, ``quick_iterations``, ``cooling_mode``
        and ``linear_rate`` as well as display options ``unit_name``,
        ``circle_color`` and ``container_color``.
    """
    job = jobs.get(job_id)
    if job is None:
        return

    # Initialize progress tracking
    job["status"] = "running"
    job["phase"] = 1
    job["phase1_progress"] = 0.0
    job["phase2_progress"] = 0.0
    now = time.time()
    job["phase1_start_time"] = now
    job["phase2_start_time"] = None
    job["phase1_elapsed"] = 0.0
    job["phase2_elapsed"] = 0.0
    job["energy"] = float("inf")
    job["success"] = None
    job["image"] = None
    # Copy simulation params into job for later time estimates
    job["iterations"] = params["iterations"]
    job["num_starts"] = params["num_starts"]
    job["quick_iterations"] = params["quick_iterations"]

    def progress_callback(
        iteration: int,
        total_iterations: int,
        current_start: int,
        total_starts: int,
        phase_i: int,
        total_phases: int,
        current_best_energy: float,
    ) -> None:
        # Safely update job progress. Use monotonic time measurements.
        j = jobs.get(job_id)
        if j is None or j.get("status") != "running":
            return
        now = time.time()
        # Phase 1 progress accumulates across starts
        if phase_i == 1:
            # Fraction of total quick screening work completed so far
            progress = (
                (current_start - 1) + (iteration / max(1, total_iterations))
            ) / max(1, total_starts)
            j["phase"] = 1
            j["phase1_progress"] = min(max(progress, 0.0), 1.0)
            j["phase1_elapsed"] = now - j["phase1_start_time"]
        else:
            # Initialize phase2 start time upon first callback in phase2
            if j.get("phase2_start_time") is None:
                j["phase2_start_time"] = now
                j["phase2_elapsed"] = 0.0
                j["phase2_progress"] = 0.0
            j["phase"] = 2
            j["phase2_progress"] = min(iteration / max(1, total_iterations), 1.0)
            j["phase2_elapsed"] = now - j["phase2_start_time"]
        # Update the current best energy
        j["energy"] = float(current_best_energy)

    try:
        # Run the optimization
        best_energy, centers = run_two_phase_optimization(
            N=params["N"],
            R_container=params["R_container"],
            R_circle=params["R_circle"],
            initial_temp=params["initial_temp"],
            iterations=params["iterations"],
            num_starts=params["num_starts"],
            quick_iterations=params["quick_iterations"],
            cooling_mode=params["cooling_mode"],
            linear_rate=params["linear_rate"],
            progress_callback=progress_callback,
        )
        # Determine success based on energy threshold and existence of solution
        success = bool(centers is not None and best_energy < SOLUTION_ENERGY_THRESHOLD)
        image_b64 = None
        if centers is not None:
            # Render final image and encode as base64
            buf: io.BytesIO = render_packing_image(
                centers,
                params["R_container"],
                params["R_circle"],
                unit_name=params.get("unit_name", "Units"),
                circle_color=params.get("circle_color", "red"),
                container_color=params.get("container_color", "blue"),
                title=f"Best Energy: {best_energy:.4f}",
            )
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        # Finalize job entry
        job.update(
            {
                "status": "completed",
                "success": success,
                "energy": float(best_energy),
                "image": image_b64,
                "phase": 2,
                "phase1_progress": 1.0,
                "phase2_progress": 1.0,
            }
        )
    except Exception as exc:
        # Mark job as errored
        job.update({"status": "error", "error": str(exc)})


#
# Job management for asynchronous simulations
#
# To support progress reporting, simulations run in background threads and
# update entries in the ``jobs`` dictionary. Each job entry contains
# progress metrics, elapsed times for each phase, and the final result.
# The keys used in each job dict are:
#   status: 'running' | 'complete' | 'error'
#   phase: 1 or 2 (current phase)
#   phase1_progress: float [0, 1]
#   phase2_progress: float [0, 1]
#   phase1_elapsed: float (seconds)
#   phase2_elapsed: float (seconds)
#   start_time: float (timestamp)
#   last_updated: float (timestamp)
#   final_energy: float
#   centers: np.ndarray or None
#   image: str (base64) or None
#   success: bool
#   error: str or None
jobs: dict[str, dict] = {}


def _simulate_job(job_id: str, params: dict) -> None:
    """Run the two‑phase optimization in a background thread.

    This function updates the global ``jobs`` dictionary with progress and
    final results. It expects ``params`` to contain all arguments
    accepted by ``run_two_phase_optimization`` except ``progress_callback``.
    """
    # Initialize job entry
    job = jobs[job_id]
    job.update({
        'status': 'running',
        'phase': 1,
        'phase1_progress': 0.0,
        'phase2_progress': 0.0,
        'phase1_elapsed': 0.0,
        'phase2_elapsed': 0.0,
        'start_time': time.time(),
        'last_updated': time.time(),
        'final_energy': None,
        'centers': None,
        'image': None,
        'success': False,
        'error': None,
    })
    # Unpack parameters for readability
    N = params['N']
    R_container = params['R_container']
    R_circle = params['R_circle']
    initial_temp = params['initial_temp']
    iterations = params['iterations']
    num_starts = params['num_starts']
    quick_iterations = params['quick_iterations']
    cooling_mode = params['cooling_mode']
    linear_rate = params['linear_rate']
    unit_name = params['unit_name']
    circle_color = params['circle_color']
    container_color = params['container_color']

    # Persist a few parameters on the job entry for later progress calculations
    job['iterations'] = iterations
    job['num_starts'] = num_starts
    job['quick_iterations'] = quick_iterations

    total_quick = num_starts * max(quick_iterations, 1)
    # Capture start times for each phase
    job['phase1_start'] = time.time()
    job['phase2_start'] = None

    def progress_callback(iteration: int, total_iterations: int, current_start: int, total_starts: int, phase: int, total_phases: int, energy: float) -> None:
        # Update progress and elapsed times per phase
        now = time.time()
        # Phase 1: quick screening across multiple starts
        if phase == 1:
            job['phase'] = 1
            completed_quick = ((current_start - 1) * total_iterations) + iteration
            job['phase1_progress'] = min(1.0, completed_quick / max(total_quick, 1))
            job['phase1_elapsed'] = now - job['phase1_start']
        elif phase == 2:
            # On first callback into phase 2, mark phase2 start
            if job['phase'] != 2:
                job['phase'] = 2
                # mark the start time for phase 2
                job['phase2_start'] = now
            job['phase2_progress'] = min(1.0, iteration / max(iterations, 1))
            if job.get('phase2_start') is not None:
                job['phase2_elapsed'] = now - job['phase2_start']
        job['last_updated'] = now

    try:
        # Run the optimization. This will update progress via callback.
        energy, centers = run_two_phase_optimization(
            N=N,
            R_container=R_container,
            R_circle=R_circle,
            initial_temp=initial_temp,
            iterations=iterations,
            num_starts=num_starts,
            quick_iterations=quick_iterations,
            cooling_mode=cooling_mode,
            linear_rate=linear_rate,
            progress_callback=progress_callback,
        )
        job['final_energy'] = energy
        job['centers'] = centers
        # Determine success
        job['success'] = bool(centers is not None and energy < SOLUTION_ENERGY_THRESHOLD)
        # Render final image
        if centers is not None:
            img_buf = render_packing_image(
                centers,
                R_container,
                R_circle,
                unit_name=unit_name,
                circle_color=circle_color,
                container_color=container_color,
                title=f"Best Energy: {energy:.4f}",
            )
            job['image'] = base64.b64encode(img_buf.getvalue()).decode('utf-8')
        else:
            job['image'] = None
        job['status'] = 'complete'
    except Exception as exc:
        job['status'] = 'error'
        job['error'] = str(exc)
    finally:
        now = time.time()
        # Ensure progress is marked complete on finish
        job['phase1_progress'] = 1.0
        job['phase2_progress'] = 1.0
        # Final elapsed times
        job['phase1_elapsed'] = (now - job['phase1_start']) if job.get('phase1_start') else 0.0
        if job.get('phase2_start'):
            job['phase2_elapsed'] = now - job['phase2_start']
        else:
            job['phase2_elapsed'] = 0.0
        job['last_updated'] = now


@app.route('/api/start', methods=['POST'])
def api_start() -> object:
    """Start a new packing simulation asynchronously.

    Returns a JSON object containing a ``job_id`` which can be used to query
    progress. The request body should contain the same parameters as
    ``/api/run``. The simulation runs in a background thread and progress
    can be polled via ``/api/progress/<job_id>``.
    """
    # Parse JSON payload
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        data = {}
    # Extract numeric parameters with caps (same logic as api_run)
    params = {
        'N': _parse_field(data, 'N', DEFAULT_N, int),
        'R_container': _parse_field(data, 'R_container', DEFAULT_R_CONTAINER, float),
        'R_circle': _parse_field(data, 'R_circle', DEFAULT_R_CIRCLE, float),
        'initial_temp': _parse_field(data, 'initial_temp', DEFAULT_TEMP, float),
        'iterations': _parse_field(data, 'iterations', min(DEFAULT_ITERATIONS, 100_000), int),
        'num_starts': _parse_field(data, 'num_starts', min(DEFAULT_NUM_STARTS, 50), int),
        'quick_iterations': _parse_field(data, 'quick_iterations', min(DEFAULT_QUICK_SCREENING_ITERATIONS, 1_000), int),
        'cooling_mode': (data.get('cooling_mode', 'log') or 'log').lower(),
        'linear_rate': _parse_field(data, 'linear_rate', 1.0, float),
        'unit_name': data.get('unit_name', 'Units') or 'Units',
        'circle_color': data.get('circle_color', 'red') or 'red',
        'container_color': data.get('container_color', 'blue') or 'blue',
    }
    if params['cooling_mode'] not in ('log', 'linear'):
        params['cooling_mode'] = 'log'
    # Create a unique job ID
    job_id = uuid.uuid4().hex
    jobs[job_id] = {}
    # Start simulation thread
    thread = threading.Thread(target=_simulate_job, args=(job_id, params), daemon=True)
    thread.start()
    return jsonify({'job_id': job_id})


@app.route('/api/progress/<job_id>', methods=['GET'])
def api_progress(job_id: str) -> object:
    """Return progress information for a running or completed job.

    If the job has completed, this endpoint also returns the final image and
    energy. If the job does not exist, returns a 404.
    """
    job = jobs.get(job_id)
    if job is None:
        return jsonify({'status': 'not_found'}), 404
    # Compute time left estimates per phase if possible
    phase1_progress = job.get('phase1_progress', 0.0)
    phase2_progress = job.get('phase2_progress', 0.0)
    phase1_elapsed = job.get('phase1_elapsed', 0.0)
    phase2_elapsed = job.get('phase2_elapsed', 0.0)
    # Estimate total time per phase based on elapsed/progress
    phase1_total_est = (phase1_elapsed / phase1_progress) if phase1_progress > 0 else None
    phase2_total_est = (phase2_elapsed / phase2_progress) if phase2_progress > 0 else None
    phase1_time_left = None
    phase2_time_left = None
    total_time_left = None
    if phase1_total_est is not None:
        phase1_time_left = max(0.0, phase1_total_est - phase1_elapsed)
    if phase2_total_est is not None:
        phase2_time_left = max(0.0, phase2_total_est - phase2_elapsed)
    # Combined time left: if phase 2 hasn't started, sum phase1_time_left and estimated phase2_total
    if job['status'] == 'complete':
        total_time_left = 0.0
    else:
        if phase2_total_est is None:
            # Estimate phase2_total based on phase1 speed if possible
            # Use time per quick iteration to estimate phase2
            if phase1_elapsed > 0 and phase1_progress > 0:
                # time per quick iteration
                time_per_quick_iter = phase1_elapsed / (phase1_progress * params['num_starts'] * max(params['quick_iterations'], 1))
                phase2_total_est = time_per_quick_iter * params['iterations']
                phase2_time_left = phase2_total_est
            # total time left is phase1_time_left + phase2_total_est
            total_time_left = (phase1_time_left or 0.0) + (phase2_total_est or 0.0)
        else:
            # Phase 2 started; total time left = phase2_time_left
            total_time_left = phase2_time_left
    # Build response
    response = {
        'status': job['status'],
        'phase': job.get('phase', 1),
        'phase1_progress': phase1_progress,
        'phase2_progress': phase2_progress,
        'phase1_time_left': phase1_time_left,
        'phase2_time_left': phase2_time_left,
        'total_time_left': total_time_left,
    }
    if job['status'] == 'complete':
        response.update({
            'success': job['success'],
            'energy': job['final_energy'],
            'image': job['image'],
        })
    elif job['status'] == 'error':
        response['error'] = job.get('error')
    return jsonify(response)


@app.route("/")
def serve_index() -> object:
    """Serve the main page (index.html) from the static front‑end."""
    # send_static_file resolves files relative to the configured ``static_folder``
    return app.send_static_file("index.html")


def _parse_field(data: dict, field: str, default, cast):
    """Extract and cast a field from the request data.

    If conversion fails, returns the provided default. This helper prevents
    exceptions when users supply malformed values.
    """
    value = data.get(field, default)
    try:
        return cast(value)
    except Exception:
        return default


@app.route("/api/run", methods=["POST"])
def api_run() -> object:
    """Run the circle packing optimization and return the result as JSON.

    The request must contain a JSON body with optional fields for simulation
    parameters. Sensible defaults are applied when fields are missing or
    malformed. The response contains the best energy and a base64‑encoded PNG
    image if a feasible packing was found.
    """
    # Parse JSON payload
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        data = {}

    # Extract numeric parameters with reasonable caps to avoid runaway CPU usage
    N = _parse_field(data, "N", DEFAULT_N, int)
    R_container = _parse_field(data, "R_container", DEFAULT_R_CONTAINER, float)
    R_circle = _parse_field(data, "R_circle", DEFAULT_R_CIRCLE, float)
    initial_temp = _parse_field(data, "initial_temp", DEFAULT_TEMP, float)
    iterations = _parse_field(
        data,
        "iterations",
        min(DEFAULT_ITERATIONS, 100_000),
        int,
    )
    num_starts = _parse_field(
        data,
        "num_starts",
        min(DEFAULT_NUM_STARTS, 50),
        int,
    )
    quick_iterations = _parse_field(
        data,
        "quick_iterations",
        min(DEFAULT_QUICK_SCREENING_ITERATIONS, 1_000),
        int,
    )

    # Mode and rate for cooling schedule
    cooling_mode = (data.get("cooling_mode", "log") or "log").lower()
    if cooling_mode not in ("log", "linear"):
        cooling_mode = "log"
    linear_rate = _parse_field(data, "linear_rate", 1.0, float)

    # Display options
    unit_name = data.get("unit_name", "Units") or "Units"
    circle_color = data.get("circle_color", "red") or "red"
    container_color = data.get("container_color", "blue") or "blue"

    # Run the optimization
    best_energy, centers = run_two_phase_optimization(
        N=N,
        R_container=R_container,
        R_circle=R_circle,
        initial_temp=initial_temp,
        iterations=iterations,
        num_starts=num_starts,
        quick_iterations=quick_iterations,
        cooling_mode=cooling_mode,
        linear_rate=linear_rate,
    )

    # Determine whether the packing was successful. A "success" occurs when
    # a feasible configuration was found (i.e. centers is not None) and the
    # resulting energy is below the solution threshold defined in packing_core.
    success = bool(centers is not None and best_energy < SOLUTION_ENERGY_THRESHOLD)

    # If we found a feasible packing, render the image and encode it. Even on
    # failure we still return the energy so the client can display it.
    if centers is not None:
        title = f"Best Energy: {best_energy:.4f}"
        image_buf: io.BytesIO = render_packing_image(
            centers,
            R_container,
            R_circle,
            unit_name=unit_name,
            circle_color=circle_color,
            container_color=container_color,
            title=title,
        )
        img_b64 = base64.b64encode(image_buf.getvalue()).decode("utf-8")
        return jsonify({"success": success, "energy": best_energy, "image": img_b64})
    else:
        return jsonify({"success": False, "energy": best_energy, "image": None})


# ---------------------------------------------------------------------------
# Asynchronous API for progress reporting
#
# Clients call POST /api/start to initiate a simulation. The response
# contains a unique job ID. The client then polls GET /api/progress/<job_id>
# every few hundred milliseconds to retrieve the current phase progress, time
# estimates and, when finished, the final image and success flag.

@app.route("/api/start", methods=["POST"])
def api_start() -> object:
    """Start a packing simulation asynchronously and return a job id.

    This endpoint parses the same parameters as ``/api/run`` but does not
    block until the computation completes. Instead it spawns a thread to
    perform the work and immediately returns a JSON object with ``job_id``.
    The client can then poll ``/api/progress/<job_id>`` for updates.
    """
    # Parse JSON payload
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        data = {}

    # Extract numeric parameters with caps similar to the synchronous API
    def parse_field(field: str, default, cast):
        value = data.get(field, default)
        try:
            return cast(value)
        except Exception:
            return default

    N = parse_field("N", DEFAULT_N, int)
    R_container = parse_field("R_container", DEFAULT_R_CONTAINER, float)
    R_circle = parse_field("R_circle", DEFAULT_R_CIRCLE, float)
    initial_temp = parse_field("initial_temp", DEFAULT_TEMP, float)
    iterations = parse_field("iterations", min(DEFAULT_ITERATIONS, 100_000), int)
    num_starts = parse_field("num_starts", min(DEFAULT_NUM_STARTS, 50), int)
    quick_iterations = parse_field(
        "quick_iterations", min(DEFAULT_QUICK_SCREENING_ITERATIONS, 1_000), int
    )
    cooling_mode = (data.get("cooling_mode", "log") or "log").lower()
    if cooling_mode not in ("log", "linear"):
        cooling_mode = "log"
    linear_rate = parse_field("linear_rate", 1.0, float)
    unit_name = data.get("unit_name", "Units") or "Units"
    circle_color = data.get("circle_color", "red") or "red"
    container_color = data.get("container_color", "blue") or "blue"

    # Create a job record and spawn a background thread
    job_id = str(uuid.uuid4())
    # Store minimal job placeholder; values are filled in _simulate_job
    jobs[job_id] = {
        "status": "initializing",
        # the rest of the fields will be set by the thread
    }
    params = {
        "N": N,
        "R_container": R_container,
        "R_circle": R_circle,
        "initial_temp": initial_temp,
        "iterations": iterations,
        "num_starts": num_starts,
        "quick_iterations": quick_iterations,
        "cooling_mode": cooling_mode,
        "linear_rate": linear_rate,
        "unit_name": unit_name,
        "circle_color": circle_color,
        "container_color": container_color,
    }
    thread = threading.Thread(
        target=_simulate_job,
        args=(job_id, params),
        daemon=True,
    )
    thread.start()
    return jsonify({"job_id": job_id})


@app.route("/api/progress/<job_id>", methods=["GET"])
def api_progress(job_id: str) -> object:
    """Return progress information for a running or completed job.

    The response JSON always includes a ``status`` key which may be
    ``running``, ``completed`` or ``error``. When running, progress values
    for each phase and estimated times are provided. When completed, the
    final energy, success flag and base64 encoded image are returned.
    """
    job = jobs.get(job_id)
    if job is None:
        return jsonify({"status": "error", "error": "invalid job id"})
    # If job errored out
    if job.get("status") == "error":
        return jsonify({"status": "error", "error": job.get("error", "unknown error")})
    # If job completed, include final results
    if job.get("status") == "completed":
        return jsonify(
            {
                "status": "completed",
                "success": job.get("success"),
                "energy": job.get("energy"),
                "image": job.get("image"),
                "phase1_progress": 1.0,
                "phase2_progress": 1.0,
                "phase1_time_left": 0,
                "phase2_time_left": 0,
                "total_time_left": 0,
            }
        )
    # Running: compute time left estimates
    phase = job.get("phase", 1)
    phase1_progress = float(job.get("phase1_progress", 0.0))
    phase2_progress = float(job.get("phase2_progress", 0.0))
    phase1_elapsed = float(job.get("phase1_elapsed", 0.0))
    phase2_elapsed = float(job.get("phase2_elapsed", 0.0))
    # Estimate time remaining for phase 1
    phase1_time_left = None
    phase2_time_left = None
    total_time_left = None
    # Estimate remaining for phase 1 if in progress
    if phase1_progress > 0 and phase1_progress < 1.0:
        # time per fraction completed
        phase1_time_left = (phase1_elapsed / phase1_progress) * (1.0 - phase1_progress)
    elif phase1_progress >= 1.0:
        phase1_time_left = 0.0
    # Estimate phase2 time left. If phase 2 has started, use its progress
    if phase >= 2:
        if phase2_progress > 0 and phase2_progress < 1.0:
            phase2_time_left = (phase2_elapsed / phase2_progress) * (1.0 - phase2_progress)
        elif phase2_progress >= 1.0:
            phase2_time_left = 0.0
        else:
            phase2_time_left = None
    else:
        # Phase 2 not started: estimate based on phase1 performance and known iteration counts
        if phase1_progress > 0:
            # quick total work units = quick_iterations * num_starts
            quick_total_work = job.get("quick_iterations", 1) * job.get("num_starts", 1)
            # work done so far in quick phase = fraction * quick_total work
            quick_work_done = phase1_progress * quick_total_work
            # average time per unit of work in phase 1
            time_per_work_unit = phase1_elapsed / quick_work_done if quick_work_done > 0 else None
            if time_per_work_unit is not None:
                phase2_total_work = job.get("iterations", 1)
                phase2_time_left = time_per_work_unit * phase2_total_work
    # Compute total time left by summing known parts
    if phase2_time_left is not None and phase1_time_left is not None:
        total_time_left = phase1_time_left + phase2_time_left
    elif phase2_time_left is not None:
        total_time_left = phase2_time_left
    elif phase1_time_left is not None:
        total_time_left = phase1_time_left

    return jsonify(
        {
            "status": "running",
            "phase": phase,
            "phase1_progress": phase1_progress,
            "phase2_progress": phase2_progress,
            "phase1_time_left": phase1_time_left,
            "phase2_time_left": phase2_time_left,
            "total_time_left": total_time_left,
            "energy": job.get("energy"),
        }
    )


if __name__ == "__main__":
    # The default port for Flask is 5000, but Render exposes port 10000. We
    # explicitly bind to 0.0.0.0 so the app is reachable over the network.
    app.run(host="0.0.0.0", port=10000)