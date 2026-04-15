# Building from Source

If you are a Windows user who just wants to run the app, stop here and use the
GitHub release instead of this guide:

1. Open the repository's Releases page.
2. Download `RisingSun-windows-x86_64-setup.exe`.
3. Run the installer.
4. Launch Rising Sun from the Start menu or desktop shortcut.

Portable fallback:

- Download `RisingSun-windows-x86_64.zip`.
- Extract it.
- Run `RisingSun.exe`.

This file is for developers who need to build from source or publish releases.

This guide covers building the Rising Sun IDOC Lookup app from source,
creating standalone executables, and Docker deployment.

## Prerequisites

- **Python 3.11+**
- **Node.js 20+** (for the frontend build)
- **Git LFS** (the TrOCR model weights are tracked with LFS)
- **Conda** (recommended) or virtualenv

### Installing Git LFS

Git LFS is required to download the 236 MB TrOCR model weights.

**Ubuntu / Debian:**
```bash
sudo apt-get install -y git-lfs
```

**macOS (Homebrew):**
```bash
brew install git-lfs
```

**Windows:**
```powershell
winget install GitHub.GitLFS
```

**Conda (any platform):**
```bash
conda install -c conda-forge git-lfs
```

## 1. Clone and Set Up

```bash
git lfs install   # one-time setup — must run BEFORE cloning
git clone https://github.com/ColinGreeley/Rising_Sun.git
cd Rising_Sun
```

If you already cloned without LFS, pull the model weights:

```bash
git lfs install
git lfs pull
```

Verify the model downloaded correctly:

```bash
ls -lh output/trocr_model_v3b/model.safetensors
# Should show ~236M, not a tiny LFS pointer file
```

## 2. Python Environment

```bash
conda create -n rising_sun python=3.11 -y
conda activate rising_sun

# Install CPU-only PyTorch (saves ~4 GB vs CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the project with web app dependencies
pip install -e '.[webapp]'
```

## 3. Build the Frontend

```bash
cd web/frontend
npm install
npm run build
cd ../..
```

This creates `web/frontend/dist/` which the backend serves as a static SPA.

## 4. Run the Dev Server

```bash
cd web/backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. Upload a housing application PDF and the app
will OCR it, extract the IDOC number, and verify it against the IDOC website.

## 5. Build a Standalone Executable (PyInstaller)

The standalone bundles Python, PyTorch, the TrOCR model, RapidOCR, and the
frontend into a single directory that runs without any dependencies.

### Automated build

```bash
bash scripts/build_standalone.sh
```

This produces `dist/RisingSun-<platform>-<arch>.tar.gz`.

### Manual build

```bash
# Ensure frontend is built first (step 3 above)

# Install PyInstaller
pip install pyinstaller

# Build
pyinstaller --noconfirm rising_sun.spec

# Test
./dist/RisingSun/RisingSun
```

The executable starts a local web server and opens your browser automatically.

### Bundle size

| Component | Size |
|---|---|
| PyTorch (CPU) | ~430 MB |
| TrOCR model weights | 236 MB |
| RapidOCR ONNX models | ~16 MB |
| Frontend + other deps | ~200 MB |
| **Total (uncompressed)** | **~1.9 GB** |
| **Compressed (.tar.gz)** | **~720 MB** |

## 6. Docker Deployment

For hosting the app on a server:

```bash
# Build and run
docker compose up --build

# Or build the image directly
docker build -t rising-sun:latest .
docker run -p 8000:8000 rising-sun:latest
```

### Production with HTTPS

1. Edit `deploy/Caddyfile` — replace `YOUR_DOMAIN` with your actual domain
2. Run `docker compose up -d`
3. Caddy automatically provisions Let's Encrypt certificates

Recommended server: **Hetzner CX22** (2 vCPU, 4 GB RAM, ~$4.50/mo).
TrOCR inference averages 164 ms per crop on CPU.

## 7. Cross-Platform Builds (GitHub Actions)

The repo includes a GitHub Actions workflow at
`.github/workflows/build-standalone.yml` that builds for both Linux and
Windows automatically on tag push and publishes the archives to a GitHub
Release:

```bash
git tag v1.0.0
git push origin v1.0.0
```

Expected release assets:

- `RisingSun-windows-x86_64-setup.exe`
- `RisingSun-windows-x86_64.zip`
- `RisingSun-linux-x86_64.tar.gz`

If a tag exists but no release appears, the workflow failed and you should check
the Actions tab for that tag run.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RISING_SUN_MODEL_DIR` | `output/trocr_model_v3b` | Path to TrOCR model weights |
| `RISING_SUN_FRONTEND_DIR` | `web/frontend/dist` | Path to built frontend files |
| `CORS_ORIGINS` | *(empty)* | Comma-separated allowed origins |

## Troubleshooting

### Model weights are a small text file

You need Git LFS. Run `git lfs install && git lfs pull`.

### `ModuleNotFoundError: No module named 'torch'`

Install PyTorch separately before the project:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### PyInstaller build includes CUDA libraries

Make sure you have CPU-only PyTorch installed. Check with:
```bash
python -c "import torch; print(torch.__version__)"
# Should end with +cpu, e.g. 2.11.0+cpu
```

### Frontend shows blank page

Build the frontend first: `cd web/frontend && npm install && npm run build`
