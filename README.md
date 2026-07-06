# ClaritySR

Local Flask web app for AI image upscaling, artifact cleanup, and face restoration.

## Features

- Real-ESRGAN, ESRGAN, BSRGAN, and GFPGAN model options
- 1x, 2x, 4x, and 8x output scale choices
- Tiled inference for large images and low-memory machines
- Optional pre-denoise filters
- Before/after slider, side-by-side zoom, download, and history
- CPU and CUDA GPU support through PyTorch

## Requirements

- Python 3.10+
- Windows, macOS, or Linux
- NVIDIA CUDA is optional but strongly recommended for large images

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA, install the PyTorch build matching your driver from the official PyTorch selector, then install the remaining requirements.

## Run

```powershell
python app.py
```

Open `http://127.0.0.1:5000`.

Model weights download automatically into `.models/` on first use.

## Configuration

Environment variables:

- `CLARITYSR_HOST`: bind host, default `127.0.0.1`
- `CLARITYSR_PORT`: bind port, default `5000`
- `CLARITYSR_DEBUG`: set `true` for Flask debug mode
- `CLARITYSR_MAX_UPLOAD_MB`: upload limit, default `30`

Example:

```powershell
$env:CLARITYSR_PORT = "8080"
python app.py
```

## Test

```powershell
python scratch/test_upscale.py
```

The test creates a dummy image and validates the main upscaling/restoration paths. First run may take time because model weights are downloaded.

## Project Layout

- `app.py`: Flask API, background task status, history, file serving
- `model.py`: model definitions, downloads, preprocessing, tiled inference
- `templates/index.html`: web UI
- `static/css/style.css`: UI styling
- `static/js/main.js`: upload flow, progress stream, viewer, history
- `scratch/test_upscale.py`: pipeline smoke test

## Runtime Data

Ignored generated folders/files:

- `.models/`: downloaded weights
- `uploads/`: source images
- `outputs/`: processed images
- `history.json`: local processing history
