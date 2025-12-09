"""
Core algorithm for the circle packing web app.

This module exposes a two‑phase simulated annealing algorithm for packing
identical circles inside a larger container circle. The goal is to minimize
overlaps and boundary violations. A helper is provided to render a diagram
of the final circle layout to an in‑memory image buffer. The implementation
is intentionally free of any GUI dependencies so that it can be safely
imported into serverless functions.

Adapted from the user's original `packing_viz7.py` script.
"""

import warnings
import matplotlib
matplotlib.use("Agg")  # Use a non‑GUI backend suitable for serverless
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from typing import Optional, Callable
from io import BytesIO

# Constants controlling the cooling schedule and energy thresholds
LOG_COOLING_CONSTANT = 0.0001
SOLUTION_ENERGY_THRESHOLD = 1.0

# Default parameters used if the client does not supply custom values.
DEFAULT_N = 21
DEFAULT_TEMP = 4000.0
DEFAULT_ITERATIONS = 3000000
DEFAULT_NUM_STARTS = 2000
DEFAULT_QUICK_SCREENING_ITERATIONS = 5000

DEFAULT_R_CONTAINER = 85000.0
DEFAULT_R_CIRCLE = 16500.0
DEFAULT_UNIT_NAME = "Units"
DEFAULT_CIRCLE_COLOR = "red"
DEFAULT_CONTAINER_COLOR = "blue"

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Attempt to set non-positive xlim on a log-scaled axis will be ignored."
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=(
        "Attempting to set identical low and high xlims makes transformation singular;"
        " automatically expanding."
    ),
)


class SimulatedAnnealingPacker:
    """Simulated annealing engine to pack circles into a container circle."""

    def __init__(self, N, R_container, R_circle, iterations, initial_temp,
                 initial_centers=None, cooling_mode="log", linear_rate=1.0):
        self.N = N
        self.R_CONTAINER = R_container
        self.R_CIRCLE = R_circle
        self.ITERATIONS = iterations
        self.INITIAL_TEMP = initial_temp

        # Choose cooling schedule
        self.cooling_mode = cooling_mode.lower()
        self.linear_rate = linear_rate

        # Derived constants
        self.D_SEPARATION = 2 * self.R_CIRCLE
        self.MAX_CENTER_DIST = self.R_CONTAINER - self.R_CIRCLE

        if initial_centers is not None:
            self.centers = np.copy(initial_centers)
        else:
            self.centers = self._initialize_random_centers()

        self.best_centers = np.copy(self.centers)
        self.best_energy = self._calculate_energy(self.centers)

        # Tracking
        self.run_iteration = 0
        self.energy_history = []
        self.temp_history = []
        self.terminate_flag = False
        self.controller = None  # kept for compatibility; unused here

    @staticmethod
    def _format_energy(n: float) -> str:
        """Return a human‑readable string for energy values."""
        if n == float('inf'):
            return "Inf"
        sign = "-" if n < 0 else ""
        n = abs(float(n))
        if n >= 1e15:
            return f"{sign}{n / 1e15:.0f}QD"
        if n >= 1e12:
            return f"{sign}{n / 1e12:.0f}T"
        if n >= 1e9:
            return f"{sign}{n / 1e9:.0f}B"
        if n >= 1e6:
            return f"{sign}{n / 1e6:.0f}M"
        if n >= 1e3:
            return f"{sign}{n / 1e3:.0f}K"
        return f"{sign}{n:.4f}"

    def _initialize_random_centers(self) -> np.ndarray:
        """Generate random starting positions within the container."""
        if self.N <= 0:
            return np.zeros((0, 2), dtype=float)

        u = np.random.rand(self.N)
        theta = 2.0 * np.pi * np.random.rand(self.N)
        r = self.MAX_CENTER_DIST * np.sqrt(u)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        centers = np.stack([x, y], axis=1)
        return centers

    def _calculate_energy(self, centers: np.ndarray) -> float:
        """Compute a penalty score for overlaps and boundary violations."""
        energy = 0.0
        # Overlap penalty
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist_sq = dx * dx + dy * dy
                required = self.D_SEPARATION ** 2
                if dist_sq < required:
                    overlap = self.D_SEPARATION - np.sqrt(dist_sq)
                    energy += 10.0 * overlap * overlap
        # Boundary penalty
        for i in range(self.N):
            cx, cy = centers[i]
            dist_sq = cx * cx + cy * cy
            if dist_sq > self.MAX_CENTER_DIST ** 2:
                violation = np.sqrt(dist_sq) - self.MAX_CENTER_DIST
                energy += 50.0 * violation * violation
        return energy

    def _generate_neighbor(self, centers: np.ndarray, temp: float) -> np.ndarray:
        """Create a neighboring state by moving one circle randomly."""
        new_centers = np.copy(centers)
        idx = random.randint(0, self.N - 1)
        max_step = temp * 0.001 * self.D_SEPARATION
        new_centers[idx, 0] += random.uniform(-max_step, max_step)
        new_centers[idx, 1] += random.uniform(-max_step, max_step)
        return new_centers

    def _temperature(self, i: int) -> float:
        """Return the current temperature at iteration i."""
        if self.cooling_mode == "linear":
            frac = min(1.0, max(0.0, i / max(1, self.ITERATIONS)))
            temp = self.INITIAL_TEMP * max(0.0, 1.0 - self.linear_rate * frac)
            return max(temp, 1e-12)
        else:
            return self.INITIAL_TEMP / (1 + LOG_COOLING_CONSTANT * np.log(max(1, i)))

    def run(
        self,
        current_start: int = 1,
        total_starts: int = 1,
        phase_i: int = 1,
        total_phases: int = 1,
        progress_callback: Optional[
            Callable[
                [int, int, int, int, int, int, float],
                None,
            ]
        ] = None,
    ) -> tuple[float, np.ndarray]:
        """Run the annealing process for a single start."""
        current_centers = self.centers
        current_energy = self.best_energy

        # For sampling the energy history occasionally
        PRINT_INTERVAL = max(1, self.ITERATIONS // 50)

        if not self.energy_history or self.energy_history[-1][0] != 0:
            self.energy_history.append((0, self.best_energy))
            self.temp_history.append((0, self._temperature(1)))

        for i in range(1, self.ITERATIONS + 1):
            self.run_iteration = i
            if self.terminate_flag:
                break
            if phase_i == 2 and self.best_energy < SOLUTION_ENERGY_THRESHOLD:
                break

            temp = self._temperature(i)
            if temp < 1e-12:
                break

            new_centers = self._generate_neighbor(current_centers, temp)
            new_energy = self._calculate_energy(new_centers)

            if new_energy < current_energy:
                current_centers = new_centers
                current_energy = new_energy
                if new_energy < self.best_energy:
                    self.best_energy = new_energy
                    self.best_centers = np.copy(new_centers)
            else:
                # Accept worse solutions with a probability dependent on temperature
                if random.random() < np.exp(-(new_energy - current_energy) / max(temp, 1e-12)):
                    current_centers = new_centers
                    current_energy = new_energy

            if i % PRINT_INTERVAL == 0 or i == self.ITERATIONS:
                # Save history for plotting or inspection
                if i not in [h[0] for h in self.energy_history]:
                    self.energy_history.append((i, self.best_energy))
                    self.temp_history.append((i, temp))
                # Report progress to callback. The callback signature is:
                #   callback(iteration, total_iterations, current_start, total_starts, phase, total_phases, energy)
                if progress_callback is not None:
                    try:
                        progress_callback(
                            i,
                            self.ITERATIONS,
                            current_start,
                            total_starts,
                            phase_i,
                            total_phases,
                            self.best_energy,
                        )
                    except Exception:
                        # Ignore errors in callback to avoid interrupting the simulation
                        pass

        return self.best_energy, self.best_centers


def run_two_phase_optimization(
    N: int = DEFAULT_N,
    R_container: float = DEFAULT_R_CONTAINER,
    R_circle: float = DEFAULT_R_CIRCLE,
    initial_temp: float = DEFAULT_TEMP,
    iterations: int = DEFAULT_ITERATIONS,
    num_starts: int = DEFAULT_NUM_STARTS,
    quick_iterations: int = DEFAULT_QUICK_SCREENING_ITERATIONS,
    cooling_mode: str = "log",
    linear_rate: float = 1.0,
    progress_callback: Optional[
        Callable[
            [int, int, int, int, int, int, float],
            None,
        ]
    ] = None,
) -> tuple[float, np.ndarray]:
    """Run two simulated annealing phases to find a good packing."""
    global_best_energy = float("inf")
    best_screening_centers = None

    # Phase 1: run quick screening on multiple random starts
    for start_i in range(1, num_starts + 1):
        packer = SimulatedAnnealingPacker(
            N, R_container, R_circle,
            quick_iterations, initial_temp,
            cooling_mode=cooling_mode,
            linear_rate=linear_rate,
        )
        e, centers = packer.run(
            current_start=start_i,
            total_starts=num_starts,
            phase_i=1,
            total_phases=2,
            progress_callback=progress_callback,
        )
        if e < global_best_energy:
            global_best_energy = e
            best_screening_centers = centers.copy()

    if best_screening_centers is None:
        return float("inf"), None

    # Phase 2: run full annealing starting from the best screening result
    packer = SimulatedAnnealingPacker(
        N, R_container, R_circle,
        iterations, initial_temp,
        initial_centers=best_screening_centers,
        cooling_mode=cooling_mode,
        linear_rate=linear_rate,
    )
    packer.best_energy = global_best_energy
    final_energy, final_centers = packer.run(
        current_start=1,
        total_starts=1,
        phase_i=2,
        total_phases=2,
        progress_callback=progress_callback,
    )
    return final_energy, final_centers


def render_packing_image(
    centers: np.ndarray,
    R_container: float,
    R_circle: float,
    unit_name: str = DEFAULT_UNIT_NAME,
    circle_color: str = DEFAULT_CIRCLE_COLOR,
    container_color: str = DEFAULT_CONTAINER_COLOR,
    title: str = "Packing Result",
) -> BytesIO:
    """Draw a packing diagram using Matplotlib and return a PNG in a BytesIO buffer."""
    fig, ax = plt.subplots(figsize=(6, 6))
    limit = R_container * 1.1 if R_container > 0 else 1.0
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal", adjustable="box")
    # Container circle
    container = patches.Circle(
        (0, 0), R_container,
        edgecolor=container_color,
        facecolor="none",
        linewidth=2.5,
        linestyle="--",
    )
    ax.add_patch(container)
    # Individual circles
    if centers is not None:
        for x, y in centers:
            circ = patches.Circle(
                (float(x), float(y)),
                R_circle,
                edgecolor="black",
                facecolor=circle_color,
                alpha=0.7,
                linewidth=0.8,
            )
            ax.add_patch(circ)
    # Mark container center
    ax.plot(0, 0, "o", color="gold", markersize=5, label="Center")
    ax.set_xlabel(f"X Distance ({unit_name})")
    ax.set_ylabel(f"Y Distance ({unit_name})")
    ax.set_title(title)
    ax.legend(loc="upper right")
    # Render into a bytes buffer
    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf