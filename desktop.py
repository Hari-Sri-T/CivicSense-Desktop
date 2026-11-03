"""Launch CivicSense in a native window, or in a browser with --browser."""

import argparse
import threading
import webbrowser
from pathlib import Path

from werkzeug.serving import make_server

from app import app
from model_runtime import runtime


def main():
    parser = argparse.ArgumentParser(description="CivicSense desktop")
    parser.add_argument("--browser", action="store_true", help="Use the default browser instead of a native window")
    parser.add_argument("--self-test", action="store_true", help="Check packaged resources and native dependencies without opening a window")
    parser.add_argument("--self-test-report", type=Path, default=Path("civicsense-check.json"), help="Write diagnostic results to this JSON file")
    parser.add_argument("--with-models", action="store_true", help="Also download/load the models and run a document question during --self-test")
    args = parser.parse_args()
    if args.with_models and not args.self_test:
        parser.error("--with-models requires --self-test")
    if args.self_test:
        from desktop_check import write_desktop_check
        write_desktop_check(args.self_test_report, args.with_models)
        return
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
                                  min_size=(900, 640), background_color="#080808", text_select=True)
            webview.start(icon=str(Path(__file__).resolve().parent / "static/civicsense-logo.png"))
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
