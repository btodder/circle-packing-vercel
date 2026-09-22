# Circle Packing

A web app that packs *N* identical circles inside a larger container circle using two-phase simulated annealing, then draws the result.

**Live:** https://circle-packing-vercel.vercel.app

## How it works

1. **Phase 1 – screening:** many short annealing runs from random starting layouts; the best candidate is kept.
2. **Phase 2 – refinement:** a long annealing run on that candidate to minimize overlaps and boundary violations.

The result comes back as a final energy score plus a PNG of the layout (rendered with Matplotlib).

In the browser you can set the number of circles, container and circle radii, initial temperature, iteration counts for both phases, cooling mode (logarithmic or linear), axis unit name, and colors. Progress is shown while the job runs.

## Tech stack

- Python, Flask, NumPy, Matplotlib
- Plain HTML/JS front end (`frontend/index.html`)
- Deployed on Vercel

## Run locally

Requires Python 3.10+.

```sh
pip install -r requirements.txt
python app.py
```

Open http://localhost:10000.

## API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/start` | Start a packing job in the background; returns a job id |
| `GET` | `/api/progress/<job_id>` | Poll progress; returns the energy and image when done |
| `POST` | `/api/run` | Run synchronously and return `{ energy, image }` |

Request body fields (all optional, defaults in `api/packing_core.py`): `N`, `R_container`, `R_circle`, `initial_temp`, `iterations`, `num_starts`, `quick_iterations`, `cooling_mode`, `linear_rate`, `unit_name`, `circle_color`, `container_color`.

## Project structure

```
app.py               Flask app: serves the front end and all /api routes (background jobs + progress)
api/packing_core.py  Simulated annealing algorithm and image rendering
api/run.py           Standalone serverless handler for POST /api/run
frontend/index.html  UI
vercel.json          Rewrites / to the front end
packing_core.py      Older copy of the algorithm (not imported by the app)
```
