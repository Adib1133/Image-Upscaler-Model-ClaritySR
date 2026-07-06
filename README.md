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
- Git LFS (for downloading model weights) – see [Installation](#installation) below

## Installation

### 1. Clone the Repository

> **⚠️ Important:** This repository uses Git Large File Storage (LFS) for model weights.
> **Do NOT download the repository as a ZIP file** – the model files will be incomplete (~1 KB pointer files instead of actual weights).

#### If you have Git LFS installed:

```bash
git clone https://github.com/Adib1133/Image-Upscaler-Model-ClaritySR.git
cd Image-Upscaler-Model-ClaritySR
git lfs pull
```

#### If you don't have Git LFS installed:

**Windows:**

- Download and install Git LFS from [git-lfs.com](https://git-lfs.com)
- Run `git lfs install` in your terminal
- Then clone the repository as shown above

**macOS:**

```bash
brew install git-lfs
git lfs install
git clone https://github.com/Adib1133/Image-Upscaler-Model-ClaritySR.git
cd Image-Upscaler-Model-ClaritySR
git lfs pull
```

**Linux (Ubuntu/Debian):**

```bash
sudo apt install git-lfs
git lfs install
git clone https://github.com/Adib1133/Image-Upscaler-Model-ClaritySR.git
cd Image-Upscaler-Model-ClaritySR
git lfs pull
```

#### Verifying the download:

After cloning, check that the model files are their actual size (not ~1 KB):

```bash
ls -lh gfpgan/weights/detection_Resnet50_Final.pth
```

The file should be approximately 104 MB. If it shows ~1 KB, run `git lfs pull` again.

### 2. Set Up Python Environment

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

Open http://127.0.0.1:5000.

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

## Troubleshooting

**Model files are only 1 KB after cloning**

This means Git LFS didn't download the actual files. Run:

```bash
git lfs install
git lfs pull
```

**"git: 'lfs' is not a git command"**

Git LFS is not installed. Follow the installation instructions above for your operating system.

**ZIP download doesn't work**

GitHub ZIP downloads don't include LFS files. You must use `git clone` as described in the installation section.
