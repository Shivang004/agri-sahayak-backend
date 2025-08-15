#!/usr/bin/env python3
"""
Startup script for Agri-Sahayak Backend
Handles initialization, knowledge base setup, and application startup
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        "fastapi", "uvicorn", "pydantic", "requests", 
        "google-generativeai", "faiss-cpu"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run: pip install -r requirements.txt")
            return False
    
    print("✅ All dependencies are available!")
    return True


def check_environment():
    """Check environment configuration"""
    print("🔧 Checking environment configuration...")
    
    # Check API keys
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key or google_key == "your_google_api_key_here":
        print("⚠️  Google API key not configured. Some features may not work.")
        print("   Please set GOOGLE_API_KEY in your .env file")
    
    print("✅ Environment check completed!")



def start_application():
    """Start the FastAPI application"""
    print("🚀 Starting Agri-Sahayak Backend...")
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    
    print(f"📍 Server will be available at: http://{host}:{port}")
    print(f"📖 API Documentation: http://{host}:{port}/docs")
    print(f"🔧 Debug mode: {debug}")
    print("\n" + "="*50)
    print("🎯 Agri-Sahayak Backend is starting...")
    print("="*50 + "\n")
    
    # Import and run the application
    try:
        from main import app
        import uvicorn
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=debug,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down Agri-Sahayak Backend...")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        sys.exit(1)


def main():
    """Main startup function"""
    print("🌾 Agri-Sahayak Backend Startup")
    print("="*40)
    
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("❌ Error: main.py not found. Please run this script from the backend directory.")
        sys.exit(1)
    
    # Run startup checks
    if not check_dependencies():
        sys.exit(1)
    
    check_environment()
    
    # Start the application
    start_application()


if __name__ == "__main__":
    main()
