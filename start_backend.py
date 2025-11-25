#!/usr/bin/env python3
"""
Standalone backend starter for Skillmotion.AI
Use this if you want to run only the backend server
"""

import os
import sys
from pathlib import Path

# Add backend directory to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    import uvicorn
    from main import app

    print("🚀 Starting Skillmotion.AI Backend Server")
    print("📍 API Documentation: http://localhost:8000/docs")
    print("🔗 Health Check: http://localhost:8000/health")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
