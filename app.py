import os
import time
import json
import uuid
import threading
import subprocess
import re
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import psutil
import torch

# Import our upscaler
from model import ImageUpscaler

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('CLARITYSR_MAX_UPLOAD_MB', '30')) * 1024 * 1024

# Prime CPU usage so first poll returns accurate value
_cpu_prime = psutil.cpu_percent(interval=None)

# Configure folders
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
OUTPUT_FOLDER = os.path.join(APP_ROOT, 'outputs')
HISTORY_FILE = os.path.join(APP_ROOT, 'history.json')
MODELS_DIR = os.path.join(APP_ROOT, '.models')
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
ALLOWED_OUTPUT_FORMATS = {'png', 'jpeg'}
ALLOWED_DENOISE_LEVELS = {'none', 'mild', 'medium', 'strong'}
ALLOWED_TARGET_SCALES = {1, 2, 4, 8}
ALLOWED_TILE_SIZES = {0, 128, 256, 512}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Initialize upscaler with absolute models path
upscaler = ImageUpscaler(models_dir=MODELS_DIR)

# Global state for tracking active task progress
# Thread-safe writing and reading
current_status = {
    "message": "Idle",
    "progress": 0,
    "active": False,
    "elapsed": 0.0,
    "eta": 0.0,          # Estimated time remaining (seconds)
    "output_filename": "",
    "input_filename": ""
}
status_lock = threading.Lock()
history_lock = threading.Lock()

# Stop event can be set to terminate the current worker.
stop_event = threading.Event()


# ==========================================
# History helpers
# ==========================================

def get_history():
    """Returns history entries whose output files still exist on disk."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with history_lock, open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []

    # Filter out entries where the output file no longer exists
    valid = []
    for item in history:
        out_file = item.get('output_file', '')
        if out_file:
            out_path = os.path.join(OUTPUT_FOLDER, out_file)
            if os.path.isfile(out_path):
                valid.append(item)
    return valid


def add_history_item(item):
    history = get_history()
    history.insert(0, item)
    _save_history(history)


def _save_history(history):
    """Persist history to disk, pruning entries for missing files."""
    pruned = []
    for item in history:
        out_file = item.get('output_file', '')
        if out_file:
            out_path = os.path.join(OUTPUT_FOLDER, out_file)
            if os.path.isfile(out_path):
                pruned.append(item)
    try:
        with history_lock, open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(pruned, f, indent=4)
    except Exception as e:
        print(f"Error saving history: {e}")


def _safe_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _validate_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            return img.size
    except Exception as exc:
        raise ValueError(f"Invalid image file: {exc}") from exc


# ==========================================
# Routes
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/system_info', methods=['GET'])
def system_info():
    """Returns CPU, RAM, and GPU diagnostics."""
    ram = psutil.virtual_memory()
    device_name = "CPU"
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)

    # Accurate CPU: use interval=0.5 for a real measurement
    cpu_usage = round(psutil.cpu_percent(interval=0.5), 1)

    # GPU info via Windows Performance Counters
    gpu_info = _get_gpu_info()

    return jsonify({
        "cpu_usage": cpu_usage,
        "ram_usage": ram.percent,
        "ram_total_gb": round(ram.total / (1024 ** 3), 1),
        "device": device_name,
        "pytorch_version": torch.__version__,
        "gpus": gpu_info
    })


def _get_gpu_info():
    """Query GPU utilization via Windows Performance Counters.
    Returns a list of dicts: [{name, label, usage_pct}]
    """
    try:
        # Step 1: get adapter names + PNP IDs to build LUID-to-name mapping
        ps_adapters = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             'Get-CimInstance Win32_VideoController | Select-Object Name, PNPDeviceID | ConvertTo-Json -Compress'],
            capture_output=True, timeout=5
        )
        adapters_raw = ps_adapters.stdout.decode('utf-8', errors='replace').strip()
        adapters_data = json.loads(adapters_raw) if adapters_raw else []
        if isinstance(adapters_data, dict):
            adapters_data = [adapters_data]

        # Filter out software/basic/virtual adapters
        virtual_keywords = ['basic display', 'microsoft', 'remote', 'virtual', 'rdp', 'vmware', 'virtualbox']
        real_adapters = [
            a for a in adapters_data
            if not any(k in (a.get('Name') or '').lower() for k in virtual_keywords)
        ]

        # Build LUID -> adapter name map using the hex LUID embedded in PNP device IDs
        # PNP format example: PCI\VEN_8086&DEV_4692&SUBSYS_...
        # LUID extraction: use ordering since we can't directly read LUID from PNP
        real_names = [a.get('Name', f'GPU {i}') for i, a in enumerate(real_adapters)]

        # Step 2: get GPU engine utilization counters (1-second sample)
        ps_gpu = subprocess.run(
            ['powershell', '-NoProfile', '-Command',
             "Get-Counter -Counter '\\GPU Engine(*)\\Utilization Percentage' -SampleInterval 1 -MaxSamples 1 "
             "| Select-Object -ExpandProperty CounterSamples "
             "| Select-Object InstanceName, CookedValue "
             "| ConvertTo-Json -Compress"],
            capture_output=True, timeout=10
        )
        gpu_raw = ps_gpu.stdout.decode('utf-8', errors='replace').strip()
        samples = json.loads(gpu_raw) if gpu_raw else []
        if isinstance(samples, dict):
            samples = [samples]

        # Aggregate 3D engine utilization per LUID (each LUID = one physical GPU adapter)
        # Instance format: pid_NNN_luid_0x00000000_0xXXXXXXXX_phys_N_eng_N_engtype_3d
        luid_3d   = {}   # LUID -> sum of 3D engine utilisation
        luid_all  = {}   # LUID -> sum of ALL engine utilisation (fallback)
        luid_order = []
        for s in samples:
            iname = s.get('InstanceName', '').lower()
            val   = float(s.get('CookedValue', 0))
            m = re.search(r'luid_(0x[0-9a-f]+_0x[0-9a-f]+)', iname)
            if not m:
                continue
            luid = m.group(1)
            if luid not in luid_all:
                luid_all[luid]  = 0.0
                luid_3d[luid]   = 0.0
                luid_order.append(luid)
            luid_all[luid] += val
            if 'engtype_3d' in iname:
                luid_3d[luid] += val

        # Only keep LUIDs that have non-zero all-engine activity or match known real adapters
        # Filter out LUIDs where all engines are 0 and we have more LUIDs than real adapters
        active_luids = [l for l in luid_order if luid_all[l] > 0 or len(luid_order) <= len(real_names)]
        # If still more LUIDs than adapters, keep only the first N
        if len(active_luids) > len(real_names) and len(real_names) > 0:
            active_luids = active_luids[:len(real_names)]

        # Build result list using 3D engine sum, the best proxy for compute load.
        result = []
        for i, luid in enumerate(active_luids):
            raw_usage = luid_3d[luid]
            usage = round(min(raw_usage, 100.0), 1)
            name = real_names[i] if i < len(real_names) else f'GPU {i}'
            name_lower = name.lower()
            igpu_keywords = ['intel', 'uhd', 'iris', 'integrated', 'radeon graphics', 'vega', 'hd graphics']
            is_igpu = any(k in name_lower for k in igpu_keywords)
            label = 'iGPU' if is_igpu else 'GPU'
            result.append({'name': name, 'label': label, 'usage_pct': usage})

        return result

    except Exception:
        return []


@app.route('/api/history', methods=['GET'])
def history():
    return jsonify(get_history())


@app.route('/api/upscale', methods=['POST'])
def start_upscale():
    """Validates parameters, saves file, and starts background thread."""
    global current_status

    # Check if a task is already running
    with status_lock:
        if current_status["active"]:
            return jsonify({"error": "An upscaling task is already in progress."}), 400

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400

    # Read and validate form parameters.
    model_key = request.form.get('model', 'realesrgan-general')
    if model_key not in upscaler.MODEL_CONFIGS:
        return jsonify({"error": "Unknown model selected."}), 400

    tile_size = _safe_int(request.form.get('tile_size'), 512)
    if tile_size not in ALLOWED_TILE_SIZES:
        return jsonify({"error": "Invalid tile size."}), 400

    denoise_level = request.form.get('denoise', 'none')
    if denoise_level not in ALLOWED_DENOISE_LEVELS:
        return jsonify({"error": "Invalid denoise level."}), 400

    target_scale = _safe_int(request.form.get('target_scale'), 4)
    if target_scale not in ALLOWED_TARGET_SCALES:
        return jsonify({"error": "Invalid target scale."}), 400

    # Secure and save input file
    file_id = str(uuid.uuid4())[:8]
    original_filename = secure_filename(file.filename)
    ext = os.path.splitext(original_filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported image type. Use PNG, JPG, JPEG, or WEBP."}), 400

    input_filename = f"{file_id}_input{ext}"
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
    file.save(input_path)

    # Output file settings
    output_format = request.form.get('format', 'png').lower()
    if output_format not in ALLOWED_OUTPUT_FORMATS:
        Path(input_path).unlink(missing_ok=True)
        return jsonify({"error": "Invalid output format."}), 400

    output_filename = f"{file_id}_upscaled.{output_format}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    # Get original image details
    try:
        width, height = _validate_image(input_path)
    except Exception as e:
        Path(input_path).unlink(missing_ok=True)
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    # Clear stop event, reset status
    stop_event.clear()

    with status_lock:
        current_status["message"] = "Initializing..."
        current_status["progress"] = 0
        current_status["active"] = True
        current_status["elapsed"] = 0.0
        current_status["eta"] = 0.0
        current_status["output_filename"] = output_filename
        current_status["input_filename"] = input_filename

    # Start upscaling in a background thread
    t = threading.Thread(
        target=upscale_worker,
        args=(input_path, output_path, model_key, tile_size, denoise_level,
              target_scale, file.filename, width, height, output_filename)
    )
    t.daemon = True
    t.start()

    return jsonify({
        "status": "started",
        "input_filename": input_filename,
        "output_filename": output_filename
    })


@app.route('/api/stop', methods=['POST'])
def stop_upscale():
    """Signals the background worker to stop at the next safe checkpoint."""
    with status_lock:
        if not current_status["active"]:
            return jsonify({"message": "No active task to stop."}), 200

    stop_event.set()
    return jsonify({"message": "Stop signal sent. Task will terminate shortly."})


def upscale_worker(input_path, output_path, model_key, tile_size, denoise_level,
                   target_scale, original_name, orig_w, orig_h, output_filename):
    """Background worker for upscaling model computation."""
    global current_status
    start_time = time.time()

    def progress_callback(msg, progress):
        now = time.time()
        elapsed = round(now - start_time, 1)

        with status_lock:
            current_status["message"] = msg
            current_status["progress"] = progress
            current_status["elapsed"] = elapsed
            current_status["eta"] = 0.0

    try:
        upscaler.upscale(
            input_image_path=input_path,
            output_image_path=output_path,
            model_key=model_key,
            tile_size=tile_size,
            denoise_level=denoise_level,
            target_scale=target_scale,
            progress_callback=progress_callback,
            stop_event=stop_event
        )

        # Get output dimensions
        with Image.open(output_path) as out_img:
            out_w, out_h = out_img.size

        elapsed_time = round(time.time() - start_time, 2)

        # Log to history
        history_item = {
            "id": output_filename.split('_')[0],
            "original_name": original_name,
            "input_file": os.path.basename(input_path),
            "output_file": output_filename,
            "original_resolution": f"{orig_w}x{orig_h}",
            "upscaled_resolution": f"{out_w}x{out_h}",
            "model": upscaler.MODEL_CONFIGS[model_key]['desc'],
            "tile_size": tile_size if tile_size > 0 else "Full Image",
            "denoise": denoise_level.capitalize(),
            "time_taken": f"{elapsed_time}s",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        add_history_item(history_item)

        with status_lock:
            current_status["message"] = "Completed successfully!"
            current_status["progress"] = 100
            current_status["active"] = False
            current_status["eta"] = 0.0

    except InterruptedError:
        with status_lock:
            current_status["message"] = "Stopped by user."
            current_status["progress"] = 100
            current_status["active"] = False
            current_status["eta"] = 0.0

    except Exception as e:
        import traceback
        traceback.print_exc()
        with status_lock:
            current_status["message"] = f"Error: {str(e)}"
            current_status["progress"] = 100
            current_status["active"] = False
            current_status["eta"] = 0.0


@app.route('/api/status', methods=['GET'])
def get_status():
    """Returns the current upscale status so the frontend can survive reloads."""
    with status_lock:
        return jsonify({
            "message": current_status["message"],
            "progress": current_status["progress"],
            "active": current_status["active"],
            "elapsed": current_status["elapsed"],
            "eta": current_status["eta"],
            "output_filename": current_status["output_filename"],
            "input_filename": current_status["input_filename"]
        })


@app.route('/api/progress', methods=['GET'])
def progress():
    """SSE endpoint streaming real-time status updates."""
    filename = request.args.get('filename', '')

    def event_stream():
        last_progress = -1
        last_message = ""
        last_elapsed = 0.0
        last_eta = 0.0

        while True:
            time.sleep(0.08)
            with status_lock:
                # Protect against stale EventSource reconnects
                if current_status["output_filename"] != filename:
                    yield f"data: {json.dumps({'message': 'Idle', 'progress': 0, 'elapsed': 0.0, 'eta': 0.0, 'active': False})}\n\n"
                    break

                msg = current_status["message"]
                prog = current_status["progress"]
                elapsed = current_status["elapsed"]
                eta = current_status["eta"]
                active = current_status["active"]

            # Yield if state changed
            if (msg != last_message or prog != last_progress
                    or elapsed != last_elapsed or eta != last_eta):
                yield f"data: {json.dumps({'message': msg, 'progress': prog, 'elapsed': elapsed, 'eta': eta, 'active': active})}\n\n"
                last_message = msg
                last_progress = prog
                last_elapsed = elapsed
                last_eta = eta

            if prog >= 100 or "Error" in msg or "Stopped" in msg:
                break

    return Response(event_stream(), mimetype="text/event-stream")


# Static file serving routes
@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


@app.route('/api/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    host = os.environ.get('CLARITYSR_HOST', '127.0.0.1')
    port = int(os.environ.get('CLARITYSR_PORT', '5000'))
    debug = os.environ.get('CLARITYSR_DEBUG', '').lower() in {'1', 'true', 'yes'}
    print(f"Starting ClaritySR on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
