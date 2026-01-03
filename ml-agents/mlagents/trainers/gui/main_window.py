"""
Simple GUI for CognitionLearn - ML-Agents with GUI
This serves as the main window that replaces the CLI interface
"""
import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtGui import QIcon, QColor, QPalette
from PyQt6.QtCore import Qt, QTimer

from mlagents.trainers.gui.settings_window import SettingsWindow

try:
    from PyQt6.QtWidgets import QApplication
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CognitionLearn")
        self.setGeometry(100, 100, 1024, 768)
        
        # Set Window Icon
        current_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(current_dir, "logo.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        self.settings_window = SettingsWindow()
        self.setCentralWidget(self.settings_window)
        
        # Hide menu bar since we don't need File/Help
        self.menuBar().hide()
        
    @staticmethod
    def apply_dark_theme(app_or_window):
        # Force Fusion style
        if isinstance(app_or_window, QApplication):
            app_or_window.setStyle("Fusion")
            
            # Global Stylesheet - The most robust way to force colors
            # We set background for QWidget (base of everything), but be careful with specificity
            app_or_window.setStyleSheet("""
                QWidget {
                    background-color: #2b2b2b;
                    color: #e0e0e0;
                    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
                    font-size: 10pt;
                }
                
                /* GroupBox specific styling */
                QGroupBox {
                    border: 1px solid #505050;
                    border-radius: 6px;
                    margin-top: 24px;
                    background-color: #323232; /* Slightly lighter than background */
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    left: 10px;
                    padding: 0 5px;
                    color: #64b5f6; /* Blue title */
                    font-weight: bold;
                }
                
                /* Inputs */
                QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                    background-color: #1e1e1e;
                    border: 1px solid #555555;
                    border-radius: 4px;
                    padding: 4px;
                    color: #ffffff;
                    min-height: 20px;
                }
                QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
                    border: 1px solid #64b5f6;
                    background-color: #252525;
                }
                
                /* Buttons */
                QPushButton {
                    background-color: #0d47a1; /* Dark blue */
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 6px 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1565c0;
                }
                QPushButton:pressed {
                    background-color: #0a3d91;
                }
                QPushButton:disabled {
                    background-color: #404040;
                    color: #808080;
                }
                
                /* ScrollArea fix */
                QScrollArea {
                    border: none;
                    background-color: transparent;
                }
                /* The widget inside the scroll area */
                QScrollArea > QWidget > QWidget {
                    background-color: #2b2b2b;
                }
                
                /* Checkbox */
                QCheckBox {
                    spacing: 8px;
                }
                QCheckBox::indicator {
                    width: 18px;
                    height: 18px;
                    border-radius: 3px;
                    border: 1px solid #666;
                    background-color: #1e1e1e;
                }
                QCheckBox::indicator:checked {
                    background-color: #64b5f6;
                    border-color: #64b5f6;
                    image: none; /* In real app use an icon, but color is enough for now */
                }
                
                /* ScrollBar */
                QScrollBar:vertical {
                    border: none;
                    background: #2b2b2b;
                    width: 10px;
                    margin: 0;
                }
                QScrollBar::handle:vertical {
                    background: #505050;
                    min-height: 20px;
                    border-radius: 5px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
            """)

        # Fallback Palette for things not covered by stylesheet (optional but good practice)
        from PyQt6.QtGui import QPalette
        palette = QPalette()
        # ... (rest of palette definition can stay as backup or be removed if CSS covers all)
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        
        if isinstance(app_or_window, QApplication):
            app_or_window.setPalette(palette)
        else:
            app_or_window.setPalette(palette)

def launch_gui():
    if not GUI_AVAILABLE:
        print("Error: PyQt6 is not installed. Please install it with: pip install PyQt6")
        return

    # Handle High DPI
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    from mlagents.trainers.gui.splash_screen import show_splash_screen
    
    app, splash = show_splash_screen()
    
    # Set Global Application Icon (Critical for Linux Taskbar)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(current_dir, "logo.png")
    if os.path.exists(icon_path):
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)
    
    # Create main window but don't show yet
    window = MainWindow()
    MainWindow.apply_dark_theme(app)
    
    # Simulate loading time or wait for splash
    QTimer.singleShot(2000, lambda: (splash.close(), window.show()))
    
    sys.exit(app.exec())

if __name__ == "__main__":
    launch_gui()
