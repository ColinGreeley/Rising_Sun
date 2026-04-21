# Rising Sun — IDOC Housing Application OCR

A local-first web application that extracts **IDOC numbers**, **applicant names**,
and **RSO (Registered Sex Offender) status** from scanned Idaho Department of
Correction housing application PDFs, then verifies them against the public IDOC
resident/client search.

Upload a PDF → the app OCRs the document, extracts the IDOC number, detects the
RSO checkbox, looks up the resident on the IDOC website, and cross-checks the
name — all in one step.

## Features

- **Multi-strategy OCR** — RapidOCR at 225 / 300 / 400 DPI with CLAHE and
  binary-threshold fallbacks for poor-quality scans
- **Fine-tuned TrOCR** — a TrOCR-small model (v3b, 61.6M params) trained on
  real housing-application handwriting for IDOC numbers and applicant names
- **Roster-backed name canonicalization** — when a trusted IDOC number is
  found, the applicant name is cross-referenced against the IDOC website
  directory and replaced with the canonical (official) name
- **RSO checkbox detection** — template-matching detector identifies the
  Registered Sex Offender checkbox across multiple form versions (V1 & V2)
  and pages, with no ML model required
- **Live IDOC verification** — candidate numbers are checked against
  `idoc.idaho.gov` and matched by name (nickname-aware, order-agnostic)
  without using the Processed Apps spreadsheet in the live web path
- **Tightened number trust** — IDOC number candidates are restricted to 5–6
  digits across all extraction paths (IDOC housing form, Jotform, high-DPI
  retry) to eliminate false positives from 7–8 digit sequences
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
git clone https://github.com/ColinGreeley/Rising_Sun.git
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
   5–6 digit IDOC number candidates; a plausibility filter rejects
   7–8 digit false positives across all extraction paths
5. **RSO Detection** — dual-template matching locates the sex-offender
   question across form versions and pages, then scores checkbox fill
   levels to determine Yes/No
6. **Website Verification** — each candidate is checked against the
   IDOC resident search; results are ranked by name similarity
7. **Name Post-processing** — when a trusted IDOC number is verified,
   the OCR-extracted name is cross-referenced against the IDOC website
   directory and canonicalized to the official name on file
8. **Name Cross-check** — applicant name from OCR (or canonicalized
   name) is compared to the IDOC database using nickname-aware,
   order-agnostic matching

## Accuracy

### Live Pipeline Validation

Full live-mode validation on the 2026 Processed Apps corpus with Processed Apps
runtime lookup disabled.

| Metric | Result |
|---|---|
| Total PDFs scanned | 542 |
| Supported 4-page applications evaluated | 315 |
| Ground-truth matches scored | 302 |
| IDOC found rate | **315/315 (100%)** |
| IDOC accuracy vs truth | **269/302 (89.1%)** |
| RSO accuracy vs truth | **286/302 (94.7%)** |
| Verification mix | **240 green / 75 yellow / 0 red** |

This run uses the same backend verification path as the web app and ranks live
database hits by OCR name similarity. It does not use the Processed Apps
spreadsheet to recover or verify candidates during runtime. Of the 33 wrong
IDOC matches in the scored set, 32 were already downgraded to yellow rather
than green.

Reproduce with:

```bash
conda activate rising_sun
python scripts/eval_live_pipeline.py \
  --years 2026 \
  --output-csv output/live_pipeline_eval_2026_v1_3_0.csv \
  --summary-json output/live_pipeline_eval_2026_v1_3_0_summary.json
```

### IDOC Number Extraction Benchmark

Offline OCR benchmark on 3,019 matched housing applications from the 2025 and
2026 corpus.

| Metric | Result |
|---|---|
| Top-1 accuracy | **~99%** |
| Method | Multi-strategy OCR + TrOCR ensemble |

The pipeline tries embedded text, region-crop OCR at 225/300/400 DPI (with
CLAHE and binary-threshold variants), and a fine-tuned TrOCR model. Regex +
digit normalization produces 5–6 digit candidates which are verified against
the IDOC website. This benchmark measures extraction quality and is not the
same as the end-to-end live validation numbers above.

### RSO Checkbox Detection

| Metric | Result |
|---|---|
| **Overall balanced accuracy** | **82.8%** |
| Overall accuracy | 94.8% |
| Sensitivity (recall for RSO = Yes) | 66.7% |
| Specificity (recall for RSO = No) | 99.0% |
| TP / TN / FP / FN | 258 / 2,605 / 27 / 129 |

**Breakdown by detection method:**

| Method | Forms | Bal. Acc. | TP | TN | FP | FN |
|---|---|---|---|---|---|---|
| Template V1 | 345 | 97.5% | 81 | 255 | 7 | 2 |
| Template V2 | 1,197 | 96.6% | 177 | 991 | 20 | 9 |
| Default (no match) | 1,477 | 50.0% | 0 | 1,359 | 0 | 118 |

When the template matcher finds the "sex offender" question text (51% of
forms), accuracy is 96–97%. The remaining 49% of forms fail to match either
template and default to "No", which is correct for the vast majority but
misses 118 true positives.

**Breakdown by year:**

| Year | Forms | Bal. Acc. |
|---|---|---|
| 2025 | 2,496 | 83.5% |
| 2026 | 523 | 80.3% |

### Applicant Name OCR

| Metric | Before (OCR only) | After (+ post-processing) |
|---|---|---|
| Exact match | 21/98 (21.4%) | **31/98 (31.6%)** |
| First + last name match | 37/98 (37.8%) | 37/98 (37.8%) |
| Any-token match | 57/98 (58.2%) | 57/98 (58.2%) |

Evaluated on a 100-document benchmark (98 supported templates). Post-processing
uses roster-backed name canonicalization: when a trusted IDOC number is found
(83/98 documents), the OCR name is cross-referenced against the IDOC website
directory and replaced with the canonical name on file. This improved exact
match by **+10 documents** (15 canonicalized, 10 net exact-match gains).

Name extraction uses a wider crop + page-text regex candidate ensemble.
Printed/digital forms perform well; handwritten names remain model-limited.
Nine alternative approaches were evaluated (Tesseract, EasyOCR, Kraken, docTR,
fine-tuned TrOCR at two dataset sizes, higher DPI, PaddleOCR) — none improved
over the RapidOCR ensemble baseline for handwritten input.

## Output Shape

Each PDF produces one JSON result containing:

- `source_pdf` — original file path
- `template` — extraction template name
- `page_count` — rendered page count
- `field_results` — flat field-by-field extraction values with confidence and metadata
- `extracted` — nested structured data assembled from field keys
- `resolved_name` — canonical applicant name (from IDOC directory when available, otherwise OCR)
- `resolved_name_source` — provenance of the resolved name (e.g. `directory_by_number_canonicalized`, `ocr`)
- `page_raw_text` — full-page OCR text for manual fallback

Batch runs also write `review.csv` listing blank text fields, low-confidence
text fields, and unresolved checkbox fields for manual review.

## Notes

- The extraction template is intentionally editable. If a region is off, adjust
  the normalized coordinates in `config/idoc_application_template.yml`.
- The corpus may contain non-IDOC PDFs. Those are classified and returned as
  unsupported documents instead of being forced through the IDOC template.

## License

Private — not currently open-source. Contact the repository owner for access.
