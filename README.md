# Rising Sun — IDOC Housing Application OCR

A local-first web application that extracts **IDOC numbers** and **applicant names**
from scanned Idaho Department of Correction housing application PDFs, then verifies
them against the public IDOC resident/client search.

Upload a PDF → the app OCRs the document, extracts the IDOC number, looks up the
resident on the IDOC website, and cross-checks the name — all in one step.

## Features

- **Multi-strategy OCR** — RapidOCR at 225 / 300 / 400 DPI with CLAHE and
  binary-threshold fallbacks for poor-quality scans
- **Fine-tuned TrOCR** — a TrOCR-small model (v3b, 61.6M params) trained on
  real housing-application handwriting for IDOC numbers and applicant names
- **RSO checkbox detection** — template-matching detector identifies the
  Registered Sex Offender checkbox across multiple form versions (V1 & V2)
  and pages, with 82.8% balanced accuracy on 3,000+ forms
- **Live IDOC verification** — candidate numbers are checked against
  `idoc.idaho.gov` and matched by name (nickname-aware, order-agnostic)
- **Green / Yellow / Red status** — instant confidence signal per document
- **Standalone executable** — download and double-click, no Python required

## Quick Start

### Windows users

Use the GitHub release. Do not build from source unless you are developing.

1. Open the [Releases](../../releases) page.
2. Download `RisingSun-windows-x86_64-setup.exe` from the latest release.
3. Run the installer.
4. Launch Rising Sun from the desktop shortcut or Start menu.
5. If Windows SmartScreen appears, click **More info** and then **Run anyway**.

The app starts a local server and opens your browser automatically.

If you prefer a portable build, use `RisingSun-windows-x86_64.zip` instead.

### Option A: Pre-built standalone (recommended)

Download the latest release from the
[Releases](../../releases) page, extract, and run:

```bash
# Linux
tar xzf RisingSun-linux-x86_64.tar.gz
./RisingSun/RisingSun

# Windows — run RisingSun-windows-x86_64-setup.exe
# Portable fallback — extract the .zip and double-click RisingSun.exe
```

The app opens a browser tab automatically.

### Option B: Run from source (developers)

```bash
# 1. Clone (includes model weights via Git LFS)
git clone https://github.com/<owner>/Rising_Sun.git
cd Rising_Sun
git lfs pull

# 2. Create environment
conda create -n rising_sun python=3.11 -y
conda activate rising_sun

# 3. Install (CPU-only PyTorch keeps it lean)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[webapp]'

# 4. Build the frontend
cd web/frontend && npm install && npm run build && cd ../..

# 5. Run
cd web/backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000 in your browser.

### Option C: Docker

```bash
docker compose up --build
```

The app will be available at your configured domain (see `deploy/Caddyfile`).

## Project Structure

```
Rising_Sun/
├── src/rising_sun/          # Core OCR extraction library
│   ├── assets/              # Template images for RSO detection
│   ├── rso_detector.py      # RSO checkbox detection via template matching
│   ├── extractor.py         # Schema-driven field extraction pipeline
│   ├── identity.py          # Name normalization, nickname mapping
│   ├── idoc_lookup.py       # Spreadsheet-based fuzzy matching
│   ├── ocr.py               # RapidOCR + Tesseract backends
│   ├── pdf.py               # PDF → image rendering (PyMuPDF)
│   └── ...                  # CLI, training, calibration modules
├── web/
│   ├── backend/main.py      # FastAPI server
│   └── frontend/            # React + Vite + Tailwind UI
├── output/trocr_model_v3b/  # Fine-tuned TrOCR weights (Git LFS)
├── config/                  # Extraction template YAML
├── scripts/                 # Training & evaluation scripts
├── launcher.py              # PyInstaller entry point
├── rising_sun.spec          # PyInstaller build spec
├── Dockerfile               # Multi-stage Docker build
└── pyproject.toml           # Package config + dependency groups
```

## Building from Source

See [BUILDING.md](BUILDING.md) for detailed instructions on:
- Building the standalone executable with PyInstaller
- Docker deployment
- Cross-platform builds via GitHub Actions

## How It Works

1. **PDF Rendering** — PyMuPDF renders page 1 at multiple DPIs
2. **Region Crop OCR** — RapidOCR reads the IDOC# field region with
   progressively aggressive image preprocessing
3. **TrOCR Inference** — fine-tuned model reads IDOC# and name crops
4. **Candidate Generation** — regex + digit normalization produces
   5–6 digit IDOC number candidates
5. **RSO Detection** — dual-template matching locates the sex-offender
   question across form versions and pages, then scores checkbox fill
   levels to determine Yes/No
6. **Website Verification** — each candidate is checked against the
   IDOC resident search; results are ranked by name similarity
7. **Name Cross-check** — applicant name from OCR is compared to the
   IDOC database using nickname-aware, order-agnostic matching

## Accuracy

Batch evaluation on 3,019 matched housing applications (2025 + 2026):
- **IDOC number extraction: 99%** top-1 accuracy
- **RSO checkbox detection: 82.8%** balanced accuracy
  - Template V1: 97.5% bal. acc. (345 forms)
  - Template V2: 96.6% bal. acc. (1,197 forms)
  - Sensitivity: 66.7% · Specificity: 99.0%

## License

Private — not currently open-source. Contact the repository owner for access.

Export the worst handwritten-name failures from a benchmark CSV for targeted relabeling:

```bash
/home/cgreeley/anaconda3/envs/rising_sun/bin/python -m rising_sun.cli export-name-failure-crops \
  output/name_ocr_research_100.csv \
  --output-dir output/name_failure_crops
```

Turn that manifest into an editable one-row-per-PDF review queue:

```bash
/home/cgreeley/anaconda3/envs/rising_sun/bin/python -m rising_sun.cli build-name-review-queue \
  output/name_failure_crops/manifest.csv
```

After marking rows as `approved` and optionally choosing a better crop variant or corrected label, convert the queue into a curated dataset bundle:

```bash
/home/cgreeley/anaconda3/envs/rising_sun/bin/python -m rising_sun.cli apply-name-review-queue \
  output/name_failure_crops/review_queue.csv \
  --bundle-dir output/name_failure_crops \
  --output-dir output/name_training_dataset_reviewed
```

## Output Shape

Each PDF produces one JSON file containing:

- `source_pdf`: original file path
- `template`: extraction template name
- `page_count`: rendered page count
- `field_results`: flat field-by-field extraction values with confidence and metadata
- `extracted`: nested structured data assembled from field keys
- `page_raw_text`: full-page OCR text for manual fallback

Each run also writes `review.csv` in the output directory. It lists blank text fields, low-confidence text fields, and unresolved/conflicting yes/no checkbox fields so you can tune the template against a batch quickly.

The corpus may contain non-IDOC PDFs in the same folder. Those are now classified and returned as unsupported documents instead of being forced through the IDOC template.

## Notes

- The template is intentionally editable. If a region is off, adjust the normalized coordinates in `config/idoc_application_template.yml`.
- `paddleocr` was installed but is not the active backend because the current runtime on this machine throws an inference error. The pipeline defaults to `rapidocr-onnxruntime`, which is stable here.
- The form has handwriting and checkbox noise, so expect iterative tuning on a representative sample before relying on fully unattended extraction.
