"""
Vercel serverless function for running the circle packing optimization.

This module uses Flask to handle HTTP requests. When deployed to Vercel, the
function will respond to POST requests with a JSON body describing the packing
parameters. It will run a two‑phase simulated annealing optimization and return
the best energy along with a base64‑encoded PNG image of the packing result.

The function defines the `app` variable, which Vercel detects to launch the
WSGI application. The route is registered at `/` so that the path of the
serverless function (e.g. `/api/run`) matches the file name only once.
"""

import base64
import io
import json
from flask import Flask, request, jsonify, make_response

from .packing_core import (
    run_two_phase_optimization,
    render_packing_image,
    DEFAULT_N,
    DEFAULT_R_CONTAINER,
    DEFAULT_R_CIRCLE,
    DEFAULT_TEMP,
    DEFAULT_ITERATIONS,
    DEFAULT_NUM_STARTS,
    DEFAULT_QUICK_SCREENING_ITERATIONS,
)


app = Flask(__name__)


def _parse_json_field(data: dict, field: str, default, cast_type):
    """Helper to extract and cast a field from JSON data with fallback."""
    value = data.get(field, default)
    try:
        return cast_type(value)
    except Exception:
        return default


@app.route("/", methods=["POST"])  # root path; file name defines the base path
def run_packing() -> object:
    """Run the packing optimization and return a JSON response."""
    try:
        data = request.get_json(force=True) or {}
    except Exception:
        data = {}

    # Parse numeric parameters with sensible defaults
    N = _parse_json_field(data, "N", DEFAULT_N, int)
    R_container = _parse_json_field(data, "R_container", DEFAULT_R_CONTAINER, float)
    R_circle = _parse_json_field(data, "R_circle", DEFAULT_R_CIRCLE, float)
    initial_temp = _parse_json_field(data, "initial_temp", DEFAULT_TEMP, float)
    iterations = _parse_json_field(
        data,
        "iterations",
        min(DEFAULT_ITERATIONS, 100_000),
        int,
    )
    num_starts = _parse_json_field(
        data,
        "num_starts",
        min(DEFAULT_NUM_STARTS, 50),
        int,
    )
    quick_iterations = _parse_json_field(
        data,
        "quick_iterations",
        min(DEFAULT_QUICK_SCREENING_ITERATIONS, 1_000),
        int,
    )
    cooling_mode = (data.get("cooling_mode", "log") or "log").lower()
    if cooling_mode not in ("log", "linear"):
        cooling_mode = "log"
    linear_rate = _parse_json_field(data, "linear_rate", 1.0, float)
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

    # Render image if solution exists
    if centers is not None:
        title = f"Best Energy: {best_energy:.4f}"
        buf: io.BytesIO = render_packing_image(
            centers,
            R_container,
            R_circle,
            unit_name=unit_name,
            circle_color=circle_color,
            container_color=container_color,
            title=title,
        )
        # Encode to base64
        img_bytes = buf.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        payload = {
            "energy": best_energy,
            "image": img_b64,
        }
    else:
        payload = {
            "energy": best_energy,
            "image": None,
        }
    response = make_response(jsonify(payload))
    # Allow cross‑origin requests for development; adjust for production as needed
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# Vercel looks for this variable to serve the application
# See: https://vercel.com/docs/functions/runtimes/python#entrypoints
app.testing = False  # Ensure we run in production mode
