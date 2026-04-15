# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Rising Sun IDOC Lookup standalone app."""

import os
import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Project root (where this spec file lives)
ROOT = Path(SPECPATH)

# Collect transformers model data that importlib.resources references
transformers_datas = collect_data_files("transformers", include_py_files=False)

# RapidOCR ONNX models + config
rapidocr_datas = collect_data_files("rapidocr_onnxruntime", include_py_files=False)

# Our bundled data files
datas = [
    # TrOCR model weights (only inference files, no checkpoints)
    (str(ROOT / "output" / "trocr_model_v3b" / "config.json"), "model"),
    (str(ROOT / "output" / "trocr_model_v3b" / "generation_config.json"), "model"),
    (str(ROOT / "output" / "trocr_model_v3b" / "model.safetensors"), "model"),
    (str(ROOT / "output" / "trocr_model_v3b" / "processor_config.json"), "model"),
    (str(ROOT / "output" / "trocr_model_v3b" / "tokenizer.json"), "model"),
    (str(ROOT / "output" / "trocr_model_v3b" / "tokenizer_config.json"), "model"),
    # Built frontend
    (str(ROOT / "web" / "frontend" / "dist"), "frontend_dist"),
    # Backend code (loaded by uvicorn as "main:app")
    (str(ROOT / "web" / "backend" / "main.py"), "backend"),
]
datas += transformers_datas
datas += rapidocr_datas

# Hidden imports that PyInstaller's analysis misses
hiddenimports = [
    # FastAPI / Starlette internals
    "uvicorn.logging",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "uvicorn.lifespan.off",
    "multipart",
    "python_multipart",
    # FastAPI / Starlette
    "fastapi",
    "fastapi.middleware",
    "fastapi.middleware.cors",
    "fastapi.staticfiles",
    "fastapi.responses",
    "starlette",
    "starlette.middleware",
    "starlette.middleware.cors",
    "starlette.responses",
    "starlette.staticfiles",
    # HTTP client
    "httpx",
    "httpcore",
    "anyio",
    "sniffio",
    "certifi",
    "h11",
    # PDF
    "pymupdf",
    # Our package
    "rising_sun",
    "rising_sun.identity",
    "rising_sun.idoc_lookup",
    "rising_sun.ocr",
    "rising_sun.pdf",
    # PyTorch CPU
    "torch",
    "torch.nn",
    "torch.nn.functional",
    # Transformers model classes used at runtime
    "transformers",
    "transformers.models.trocr",
    "transformers.models.vision_encoder_decoder",
    # lxml
    "lxml.html",
    "lxml._elementpath",
    # Image processing
    "cv2",
    "PIL",
    "numpy",
]
hiddenimports += collect_submodules("rapidocr_onnxruntime")
hiddenimports += collect_submodules("fastapi")
hiddenimports += collect_submodules("starlette")
hiddenimports += collect_submodules("transformers.generation")
hiddenimports += collect_submodules("transformers.models.roberta")
hiddenimports += collect_submodules("transformers.models.trocr")
hiddenimports += collect_submodules("transformers.models.vision_encoder_decoder")
hiddenimports += collect_submodules("transformers.models.vit")

a = Analysis(
    [str(ROOT / "launcher.py")],
    pathex=[
        str(ROOT / "src"),          # rising_sun package
        str(ROOT / "web" / "backend"),  # main.py
    ],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "pyi_rth_torch_cpu.py")],
    excludes=[
        "triton",
        "apex",
        "IPython",
        "jupyter",
        "notebook",
        "tensorboard",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Strip CUDA / GPU binary libraries to keep the bundle CPU-only (~4GB savings)
_CUDA_EXCLUDE_PATTERNS = (
    "libtorch_cuda",
    "libtorch_nvshmem",
    "libcuda",
    "libcudss",
    "libc10_cuda",
    "libnvrtc",
    "libnvToolsExt",
    "libnccl",
    "libcublas",
    "libcufft",
    "libcurand",
    "libcusolver",
    "libcusparse",
    "libcudnn",
    "libnvJitLink",
    "libnvshmem",
    "libonnxruntime_providers_cuda",
    "libonnxruntime_providers_tensorrt",
    "nvidia/",
)
a.binaries = [
    b for b in a.binaries
    if not any(pat in b[0] or pat in str(b[1]) for pat in _CUDA_EXCLUDE_PATTERNS)
]

# Also strip pyarrow (pulled in by pandas/datasets but unused)
a.binaries = [b for b in a.binaries if "pyarrow" not in b[0]]

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RisingSun",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Show console so user sees "Starting server..."
    icon=None,      # Add an .ico file here if you want a custom icon
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="RisingSun",
)
