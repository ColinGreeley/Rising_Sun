"""Rising Sun IDOC Lookup — standalone launcher.

This is the entry point for the PyInstaller-bundled executable.
It starts the FastAPI server and opens the user's browser.
"""

from __future__ import annotations

import os
import sys
import socket
import threading
import time
import webbrowser


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_server(host: str, port: int, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


def main() -> None:
    # Determine base directory — PyInstaller sets sys._MEIPASS for bundled data
    if getattr(sys, "frozen", False):
        bundle_dir = sys._MEIPASS
    else:
        bundle_dir = os.path.dirname(os.path.abspath(__file__))

    # Point the backend at bundled model + frontend
    os.environ["RISING_SUN_MODEL_DIR"] = os.path.join(bundle_dir, "model")
    os.environ["RISING_SUN_FRONTEND_DIR"] = os.path.join(bundle_dir, "frontend_dist")
    os.environ["RISING_SUN_RAPIDOCR_USE_CUDA"] = "0"

    # Add backend directory to path so uvicorn can find main:app
    backend_dir = os.path.join(bundle_dir, "backend")
    if backend_dir not in sys.path:
        sys.path.insert(0, backend_dir)

    port = _find_free_port()
    host = "127.0.0.1"
    url = f"http://{host}:{port}"

    print("=" * 50)
    print("  Rising Sun IDOC Lookup")
    print("=" * 50)
    print()
    print(f"  Starting server on {url} ...")
    print()
    print("  DO NOT CLOSE THIS WINDOW while using the app.")
    print("  Press Ctrl+C to shut down.")
    print()

    # Open browser once server is ready
    def _open_browser():
        if _wait_for_server(host, port):
            webbrowser.open(url)
        else:
            print("  WARNING: Server did not start within 30 seconds.")

    threading.Thread(target=_open_browser, daemon=True).start()

    # Start uvicorn
    import uvicorn

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
