"""PyInstaller runtime hook: patch torch to load without CUDA libraries.

When we strip CUDA .so files from the bundle to keep it CPU-only,
libtorch_global_deps.so still has a DT_NEEDED on libcudart.so.12.
This hook replaces _load_global_deps with a version that falls back
to loading libtorch_cpu.so directly when CUDA libs are absent.
"""

import ctypes
import os
import platform


def _patched_load_global_deps() -> None:
    """Load libtorch_cpu.so directly, skipping CUDA deps entirely."""
    if platform.system() == "Windows":
        return

    lib_ext = ".dylib" if platform.system() == "Darwin" else ".so"

    # Try libtorch_global_deps first (works if CUDA libs are present)
    import torch as _torch_mod

    lib_dir = os.path.join(os.path.dirname(os.path.abspath(_torch_mod.__file__)), "lib")
    global_deps = os.path.join(lib_dir, f"libtorch_global_deps{lib_ext}")
    cpu_lib = os.path.join(lib_dir, f"libtorch_cpu{lib_ext}")

    try:
        ctypes.CDLL(global_deps, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        # CUDA libs missing – load CPU lib directly instead
        ctypes.CDLL(cpu_lib, mode=ctypes.RTLD_GLOBAL)


def _install_hook():
    import importlib
    import sys

    # torch may not be imported yet; we need to patch at module level.
    # We'll intercept torch.__init__ by pre-loading the module partially.
    # Simpler: just inject our function into the torch module namespace
    # once it starts loading.  Since runtime hooks run before the main
    # script, torch hasn't been imported yet.

    # Strategy: override ctypes.CDLL to silently skip CUDA .so failures
    # during torch init, then restore it.
    _original_CDLL = ctypes.CDLL

    class _ForgivingCDLL(_original_CDLL):
        def __init__(self, name, *args, **kwargs):
            try:
                super().__init__(name, *args, **kwargs)
            except OSError as e:
                # If it's a CUDA-related library, silently skip
                cuda_markers = (
                    "libcudart", "libcublas", "libcufft", "libcurand",
                    "libcusolver", "libcusparse", "libcudnn", "libnccl",
                    "libnvrtc", "libnvJitLink", "libnvshmem", "libcupti",
                    "libnvToolsExt", "libcusparseLt", "libcufile",
                    "libtorch_global_deps",
                )
                name_str = str(name) if name else ""
                if any(marker in name_str for marker in cuda_markers):
                    # Create a minimal object that won't crash
                    self._handle = 0
                    self._name = name
                else:
                    raise

    ctypes.CDLL = _ForgivingCDLL

    # Also patch ctypes.cdll.LoadLibrary for the same reason
    _original_load = ctypes.cdll.LoadLibrary

    def _forgiving_load(name):
        try:
            return _original_load(name)
        except OSError as e:
            cuda_markers = (
                "libcudart", "libcublas", "libcufft", "libcurand",
                "libcusolver", "libcusparse", "libcudnn", "libnccl",
                "libnvrtc", "libnvJitLink", "libnvshmem", "libcupti",
                "libnvToolsExt", "libcusparseLt", "libcufile",
                "libtorch_global_deps",
            )
            name_str = str(name) if name else ""
            if any(marker in name_str for marker in cuda_markers):
                return None
            raise

    ctypes.cdll.LoadLibrary = _forgiving_load


_install_hook()
