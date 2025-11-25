#!/usr/bin/env python3
"""
Development server starter for Skillmotion.AI
Starts both frontend and backend with proper configuration
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
import socket
from pathlib import Path


def find_free_port(start_port=8000):
    """Find a free port starting from start_port"""
    port = start_port
    while port < start_port + 100:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            port += 1
    return None


def start_backend():
    """Start the FastAPI backend"""
    backend_port = find_free_port(8000)
    if not backend_port:
        print("❌ Could not find free port for backend")
        return

    print(f"🚀 Starting backend on port {backend_port}")

    # Set environment variables
    os.environ['BACKEND_PORT'] = str(backend_port)

    # Add backend to path
    backend_path = os.path.abspath("backend")
    sys.path.insert(0, backend_path)

    try:
        os.chdir("backend")
        subprocess.run([
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "127.0.0.1",
            "--port", str(backend_port),
            "--reload"
        ])
    except KeyboardInterrupt:
        pass
    finally:
        os.chdir("..")


def start_frontend():
    """Start the frontend server"""
    frontend_port = find_free_port(3000)
    if not frontend_port:
        print("❌ Could not find free port for frontend")
        return

    print(f"🌐 Starting frontend on port {frontend_port}")

    # Create frontend directory and copy index.html
    frontend_dir = Path("frontend")
    frontend_dir.mkdir(exist_ok=True)

    # Copy and modify index.html for development
    with open("index.html", "r") as f:
        html_content = f.read()

    # Get backend port from environment or use default
    backend_port = os.environ.get('BACKEND_PORT', '8000')

    # Update API endpoints
    html_content = html_content.replace(
        "fetch('/api/chat'",
        f"fetch('http://localhost:{backend_port}/api/chat'"
    )

    with open(frontend_dir / "index.html", "w") as f:
        f.write(html_content)

    try:
        os.chdir("frontend")
        subprocess.run([
            sys.executable, "-m", "http.server", str(frontend_port),
            "--bind", "127.0.0.1"
        ])
    except KeyboardInterrupt:
        pass
    finally:
        os.chdir("..")


def main():
    """Main function"""
    print("🚀 Starting Skillmotion.AI Development Servers")
    print("=" * 50)

    try:
        # Start backend in thread
        backend_thread = threading.Thread(target=start_backend, daemon=True)
        backend_thread.start()

        # Wait a bit then start frontend
        time.sleep(3)
        start_frontend()

    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")


if __name__ == "__main__":
    main()
