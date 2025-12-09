import base64, io
from flask import Flask, request, jsonify
from api.packing_core import run_two_phase_optimization, render_packing_image, DEFAULT_N, DEFAULT_R_CONTAINER, DEFAULT_R_CIRCLE, DEFAULT_TEMP, DEFAULT_ITERATIONS, DEFAULT_NUM_STARTS, DEFAULT_QUICK_SCREENING_ITERATIONS

app = Flask(__name__, static_folder='frontend', static_url_path='')

@app.route('/')
def index():
    return app.send_static_file('index.html')

def parse(data, key, default, cast):
    try:
        return cast(data.get(key, default))
    except Exception:
        return default

@app.route('/api/run', methods=['POST'])
def api_run():
    data = request.get_json(force=True) or {}
    N = parse(data, 'N', DEFAULT_N, int)
    R_container = parse(data, 'R_container', DEFAULT_R_CONTAINER, float)
    R_circle = parse(data, 'R_circle', DEFAULT_R_CIRCLE, float)
    initial_temp = parse(data, 'initial_temp', DEFAULT_TEMP, float)
    iterations = parse(data, 'iterations', min(DEFAULT_ITERATIONS, 100000), int)
    num_starts = parse(data, 'num_starts', min(DEFAULT_NUM_STARTS, 50), int)
    quick_iterations = parse(data, 'quick_iterations', min(DEFAULT_QUICK_SCREENING_ITERATIONS, 1000), int)
    cooling_mode = (data.get('cooling_mode', 'log') or 'log').lower()
    if cooling_mode not in ('log','linear'):
        cooling_mode = 'log'
    linear_rate = parse(data, 'linear_rate', 1.0, float)
    unit_name = data.get('unit_name', 'Units') or 'Units'
    circle_color = data.get('circle_color', 'red') or 'red'
    container_color = data.get('container_color', 'blue') or 'blue'

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
    if centers is None:
        return jsonify({'energy': best_energy, 'image': None})
    buf = render_packing_image(
        centers,
        R_container,
        R_circle,
        unit_name=unit_name,
        circle_color=circle_color,
        container_color=container_color,
        title=f'Best Energy: {best_energy:.4f}',
    )
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return jsonify({'energy': best_energy, 'image': img_b64})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
