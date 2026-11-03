"""Launch CivicSense in a native window, or in a browser with --browser."""

import argparse
import threading
import webbrowser

from werkzeug.serving import make_server

from app import app
from model_runtime import runtime


def main():
    parser = argparse.ArgumentParser(description="CivicSense desktop")
    parser.add_argument("--browser", action="store_true", help="Use the default browser instead of a native window")
    args = parser.parse_args()
    server = make_server("127.0.0.1", 0, app, threaded=True)
    address = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    runtime.start()
    try:
        if args.browser:
            print(f"CivicSense: {address} (Ctrl+C to stop)")
            webbrowser.open(address)
            thread.join()
        else:
            import webview

            webview.create_window("CivicSense", address, width=1360, height=900,
                                  min_size=(900, 640), background_color="#101116", text_select=True)
            webview.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
