"""
Environment setup utility for LineLogic project.
This script helps manage virtual environment paths.
"""

import sys
import os

def setup_env_path(env_name="lov10-env310"):
    """
    Add virtual environment to Python path.
    
    Args:
        env_name: Name of the virtual environment folder
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    env_path = os.path.join(project_root, "envs", env_name, "Lib", "site-packages")
    
    if os.path.exists(env_path):
        if env_path not in sys.path:
            sys.path.insert(0, env_path)
            print(f"✅ Added {env_name} to Python path: {env_path}")
        else:
            print(f"ℹ️ {env_name} already in Python path")
        return True
    else:
        print(f"❌ Environment not found: {env_path}")
        return False

def list_available_envs():
    """List all available virtual environments."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    envs_dir = os.path.join(project_root, "envs")
    
    if not os.path.exists(envs_dir):
        print("❌ No envs directory found")
        return []
    
    envs = [d for d in os.listdir(envs_dir) if os.path.isdir(os.path.join(envs_dir, d))]
    print("📁 Available environments:")
    for env in envs:
        print(f"  - {env}")
    return envs

if __name__ == "__main__":
    print("🔧 LineLogic Environment Setup")
    print("=" * 40)
    
    # List available environments
    envs = list_available_envs()
    
    if envs:
        # Try to setup default environment
        if setup_env_path("lov10-env310"):
            print("\n✅ Environment setup complete!")
        else:
            print(f"\n⚠️ Default environment not found. Available: {envs}")
            if envs:
                print(f"Try: setup_env_path('{envs[0]}')")
    else:
        print("\n❌ No environments found in envs/ directory") 