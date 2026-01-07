"""
Advanced FNOL Claims Intelligence System
Main Application Entry Point
Admiral Group - Motor Insurance Claims Analytics
"""

from pathlib import Path
from shiny import App
from app_ui import app_ui
from server import server

# Get the path to the directory containing 'shiny_app' (the repo root)
# root_path = Path(__file__).resolve().parent.parent
# sys.path.append(str(root_path))

# Define the path to your www directory
www_dir = Path(__file__).parent / "www"

# Create the Shiny application
app = App(app_ui, server, static_assets=www_dir)

if __name__ == "__main__":
    app.run()