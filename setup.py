#!/usr/bin/env python3
"""
Setup script for Advanced Computer Vision Perception System
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    try:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        'cv2', 'numpy', 'torch', 'torchvision', 
        'PIL', 'matplotlib', 'ultralytics'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("✅ All dependencies are available!")
        return True

def main():
    """Main setup function"""
    print("🚗 Advanced Computer Vision Perception System Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("final_dash.py"):
        print("❌ Please run this script from the project root directory")
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Setup incomplete. Please install missing packages manually.")
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nTo run the system:")
    print("  python final_dash.py")
    print("\nFor individual components:")
    print("  python lane_detector.py")
    print("  python lane_hybrid.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
