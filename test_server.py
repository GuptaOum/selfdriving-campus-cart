import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        html = b"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Test Server</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #1a1a2e; color: white; }
                h1 { color: #00d4ff; font-size: 2.5em; }
                p { font-size: 1.2em; color: #a0a0c0; }
                .box { background: #16213e; border-radius: 12px; padding: 30px; display: inline-block; margin-top: 20px; }
            </style>
        </head>
        <body>
            <div class="box">
                <h1>It works!</h1>
                <p>Your mobile is connected to the server.</p>
                <p>Network is working correctly.</p>
            </div>
        </body>
        </html>
        """
        self.wfile.write(html)

    def log_message(self, format, *args):
        print(f"[REQUEST] {self.address_string()} -> {args[0]}")

if __name__ == "__main__":
    host = "0.0.0.0"
    port = 5000
    server = HTTPServer((host, port), Handler)
    server.allow_reuse_address = True

    # Run server in a background thread so Ctrl+C is caught immediately
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()

    print(f"Server running on http://{host}:{port}")
    print("Find your local IP with: ipconfig (Windows) or hostname -I (Linux)")
    print("Then open  http://<your-ip>:5000  on your mobile browser")
    print("Press Ctrl+C to stop.")

    try:
        thread.join()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()
        server.server_close()
        print("Server stopped.")
        sys.exit(0)
