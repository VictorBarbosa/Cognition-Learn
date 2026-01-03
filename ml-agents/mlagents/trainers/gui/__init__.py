"""
GUI module for CognitionLearn
Contains the main window and splash screen functionality
"""
try:
    from .main_window import MainWindow, launch_gui
except ImportError:
    # Fallback or dummy if dependencies are missing during install
    def launch_gui():
        print("GUI dependencies (PyQt6) not found.")

__all__ = ['MainWindow', 'launch_gui']
