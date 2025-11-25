#!/usr/bin/env python3
"""
Skillmotion.AI - AI-Powered Skill Development Assistant
Run this file to start the complete application
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
import shutil
import socket
from pathlib import Path


def find_free_port(start_port=8000):
    """Find a free port starting from start_port"""
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    return None


def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import langchain
        import groq
        import cohere
        import edge_tts
        import dotenv
        print("✅ All dependencies are available")
        return True
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e}")
        return install_dependencies()


def setup_env_file():
    """Setup .env file with user input"""
    env_path = Path(".env")
    env_example_path = Path(".env.example")

    if env_path.exists():
        # Check if API keys are set
        with open(env_path) as f:
            content = f.read()

        if "your_groq_api_key_here" not in content and "your_cohere_api_key_here" not in content:
            print("✅ Environment file is configured")
            return True

    # Create .env from template
    if env_example_path.exists():
        shutil.copy(env_example_path, env_path)
        print("📝 Created .env file from template")
    else:
        # Create basic .env file
        env_content = """# Skillmotion.AI Environment Configuration
GROQ_API_KEY=your_groq_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
APP_NAME=Skillmotion.AI
DEBUG=True
HOST=127.0.0.1
PORT=8000
"""
        with open(env_path, 'w') as f:
            f.write(env_content)
        print("📝 Created .env file")

    print("\n" + "=" * 60)
    print("🔑 API KEYS SETUP REQUIRED")
    print("=" * 60)
    print("You need free API keys from:")
    print("1. Groq (Free): https://console.groq.com/")
    print("2. Cohere (Free): https://dashboard.cohere.ai/")
    print("\nBoth services offer generous free tiers!")
    print("=" * 60)

    # Get API keys from user
    groq_key = input("\n🔹 Enter your Groq API key (or press Enter to skip): ").strip()
    cohere_key = input("🔹 Enter your Cohere API key (or press Enter to skip): ").strip()

    if not groq_key or not cohere_key:
        print("\n⚠️  You can add API keys later by editing the .env file")
        print("The application will start but AI features won't work without keys.")
        input("Press Enter to continue...")
        return False

    # Update .env file with API keys
    with open(env_path, 'r') as f:
        content = f.read()

    content = content.replace('your_groq_api_key_here', groq_key)
    content = content.replace('your_cohere_api_key_here', cohere_key)

    with open(env_path, 'w') as f:
        f.write(content)

    print("✅ API keys configured successfully!")
    return True


def start_backend(port=8000):
    """Start the FastAPI backend server"""
    print(f"🚀 Starting backend server on port {port}...")

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Add backend to Python path
    backend_path = os.path.abspath("backend")
    if backend_path not in sys.path:
        sys.path.insert(0, backend_path)

    # Set PYTHONPATH environment variable
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = backend_path + os.pathsep + current_pythonpath

    # Change to backend directory
    original_cwd = os.getcwd()
    os.chdir("backend")

    try:
        # Import and run the app
        import uvicorn
        from main import app

        print(f"✅ Backend starting on http://localhost:{port}")
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=port,
            reload=False,
            log_level="warning"  # Reduce log noise
        )
    except Exception as e:
        print(f"❌ Backend failed to start: {e}")
        print("Make sure your API keys are correctly set in .env file")
    finally:
        os.chdir(original_cwd)


def open_browser(backend_port=8000):
    """Open browser after servers start"""
    time.sleep(4)  # Wait for servers to start
    try:
        print("🌐 Opening browser...")
        webbrowser.open(f"http://localhost:{backend_port}")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print(f"Please open http://localhost:{backend_port} manually")


def test_backend(backend_port=8000):
    """Test if backend is working"""
    time.sleep(3)
    try:
        import requests
        response = requests.get(f"http://localhost:{backend_port}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Backend is running correctly")
            data = response.json()
            if data.get('ai_enabled'):
                print("✅ AI features are enabled")
            else:
                print("⚠️  AI features disabled - check API keys")
        else:
            print("⚠️  Backend responded but may have issues")
    except Exception as e:
        print("⚠️  Backend health check failed - API features may not work")
        print("Check your API keys in .env file")


def main():
    """Main function to run the application"""
    print("=" * 60)
    print("🚀 SKILLMOTION.AI - AI-POWERED SKILL DEVELOPMENT")
    print("=" * 60)

    # Check if we're in the right directory
    if not Path("frontend/index.html").exists() and not Path("index.html").exists():
        print("❌ Frontend files not found. Please run this script from the project root directory.")
        return

    # Check and install dependencies
    if not check_dependencies():
        print("❌ Failed to setup dependencies")
        return

    # Setup environment file and API keys
    api_keys_configured = setup_env_file()

    # Create static directory
    os.makedirs("static/audio", exist_ok=True)

    # Find free port for backend
    backend_port = find_free_port(8000)

    if not backend_port:
        print("❌ Could not find free port for backend")
        return

    print(f"\n📋 Starting application...")
    print(f"🔗 Backend API: http://localhost:{backend_port}")
    print(f"🌐 Frontend: http://localhost:{backend_port}")

    if not api_keys_configured:
        print("⚠️  AI features disabled - add API keys to .env file")

    print("\n💡 Press Ctrl+C to stop the server")
    print("=" * 60)

    try:
        # Start browser opener in a separate thread
        browser_thread = threading.Thread(target=open_browser, args=(backend_port,), daemon=True)
        browser_thread.start()

        # Start backend health check
        health_thread = threading.Thread(target=test_backend, args=(backend_port,), daemon=True)
        health_thread.start()

        # Start backend (this will block)
        start_backend(backend_port)

    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        print("Thank you for using Skillmotion.AI!")
    except Exception as e:
        print(f"\n❌ Application error: {e}")


if __name__ == "__main__":
    main()
