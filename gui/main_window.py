"""
Simple GUI for CognitionLearn - ML-Agents with GUI
This serves as the main window that replaces the CLI interface
"""
import sys
import os
import yaml
import subprocess
import tempfile
import threading
from typing import Optional, List, Dict, Any
from mlagents_envs import logging_util

logger = logging_util.get_logger(__name__)

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout,
        QHBoxLayout, QPushButton, QTextEdit, QLabel,
        QTabWidget, QFileDialog, QComboBox, QGroupBox,
        QSplashScreen, QScrollArea, QFormLayout,
        QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit,
        QStackedWidget, QRadioButton, QSlider
    )
    from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt6.QtGui import QFont, QIcon, QColor, QPixmap, QPainter, QPen, QBrush
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


class TrainingConfigWidget(QWidget):
    """Widget for configuring training parameters"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Configuration file selection
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout()
        
        config_h_layout = QHBoxLayout()
        config_h_layout.addWidget(QLabel("Config File:"))
        self.config_path_edit = QTextEdit()
        self.config_path_edit.setMaximumHeight(30)
        config_h_layout.addWidget(self.config_path_edit)
        
        self.browse_config_btn = QPushButton("Browse...")
        self.browse_config_btn.clicked.connect(self.browse_config)
        config_h_layout.addWidget(self.browse_config_btn)
        
        config_layout.addLayout(config_h_layout)
        config_group.setLayout(config_layout)
        layout.addWidget(config_group)
        
        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QHBoxLayout()
        
        env_layout.addWidget(QLabel("Environment Path:"))
        self.env_path_edit = QTextEdit()
        self.env_path_edit.setMaximumHeight(30)
        env_layout.addWidget(self.env_path_edit)
        
        self.browse_env_btn = QPushButton("Browse...")
        self.browse_env_btn.clicked.connect(self.browse_env)
        env_layout.addWidget(self.browse_env_btn)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group)
        
        # Run options
        options_group = QGroupBox("Run Options")
        options_layout = QVBoxLayout()
        
        options_h_layout = QHBoxLayout()
        options_h_layout.addWidget(QLabel("Run ID:"))
        self.run_id_edit = QTextEdit()
        self.run_id_edit.setMaximumHeight(30)
        self.run_id_edit.setText("ppo")
        options_h_layout.addWidget(self.run_id_edit)
        
        options_h_layout.addWidget(QLabel("Seed:"))
        self.seed_edit = QTextEdit()
        self.seed_edit.setMaximumHeight(30)
        self.seed_edit.setText("12345")
        options_h_layout.addWidget(self.seed_edit)
        
        options_layout.addLayout(options_h_layout)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def browse_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Configuration File", "", "YAML Files (*.yaml *.yml)"
        )
        if file_path:
            self.config_path_edit.setText(file_path)
    
    def browse_env(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Unity Environment", "", "All Files (*)"
        )
        if file_path:
            self.env_path_edit.setText(file_path)


class TrainingControlWidget(QWidget):
    """Widget for controlling training process"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Training controls
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Training")
        self.start_btn.setObjectName("start_btn")
        self.start_btn.clicked.connect(self.start_training)
        control_layout.addWidget(self.start_btn)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setObjectName("pause_btn")
        self.pause_btn.clicked.connect(self.pause_training)
        control_layout.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.clicked.connect(self.stop_training)
        control_layout.addWidget(self.stop_btn)
        
        layout.addLayout(control_layout)
        
        # Progress and status
        status_group = QGroupBox("Training Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready to start training...")
        self.status_label.setStyleSheet("font-weight: bold; color: #64B5F6; padding: 5px;")
        status_layout.addWidget(self.status_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def start_training(self):
        self.status_label.setText("Training in progress...")
        self.status_label.setStyleSheet("font-weight: bold; color: #81C784; padding: 5px;")  # Light green
        print("Starting training...")

    def pause_training(self):
        self.status_label.setText("Training paused")
        self.status_label.setStyleSheet("font-weight: bold; color: #FFB74D; padding: 5px;")  # Light orange
        print("Pausing training...")

    def stop_training(self):
        self.status_label.setText("Training stopped")
        self.status_label.setStyleSheet("font-weight: bold; color: #E57373; padding: 5px;")  # Light red
        print("Stopping training...")


# Default Hyperparameters for each algorithm


ALGO_DEFAULTS = {
    "ppo": {
        "batch_size": 2048, "buffer_size": 500000, "learning_rate": 0.0005,
        "beta": 0.002, "epsilon": 0.2, "lambd": 0.95, "num_epoch": 10,
        "learning_rate_schedule": "linear",
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "memory": {
                "memory_size": 256,
                "sequence_length": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.99,
                "strength": 1.0,
                "network_settings": {
                    "normalize": True,
                    "hidden_units": 256,
                    "num_layers": 2
                }
            }
        }
    },
    "sac": {
        "learning_rate": 0.0005, "learning_rate_schedule": "linear",
        "batch_size": 256, "buffer_size": 1000000, "buffer_init_steps": 20000,
        "tau": 0.005, "steps_per_update": 1.0, "save_replay_buffer": True, "init_entcoef": 0.1,
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "memory": {
                "memory_size": 256,
                "sequence_length": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.99,
                "strength": 1.0,
                "network_settings": {
                    "normalize": True,
                    "hidden_units": 256,
                    "num_layers": 2
                }
            }
        }
    },
    "td3": {
        "learning_rate": 0.0005, "learning_rate_schedule": "linear",
        "batch_size": 256, "buffer_size": 1000000, "buffer_init_steps": 20000,
        "tau": 0.005, "steps_per_update": 1.0, "save_replay_buffer": True, "policy_delay": 2,
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "memory": {
                "memory_size": 256,
                "sequence_length": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.99,
                "strength": 1.0,
                "network_settings": {
                    "normalize": True,
                    "hidden_units": 256,
                    "num_layers": 2
                }
            }
        }
    },
    "tdsac": {
        "learning_rate": 0.0005, "learning_rate_schedule": "linear",
        "batch_size": 256, "buffer_size": 1000000, "buffer_init_steps": 20000,
        "tau": 0.005, "steps_per_update": 1.0, "save_replay_buffer": True, "init_entcoef": 0.1,
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "memory": {
                "memory_size": 256,
                "sequence_length": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.99,
                "strength": 1.0,
                "network_settings": {
                    "normalize": True,
                    "hidden_units": 256,
                    "num_layers": 2
                }
            }
        }
    },
    "tqc": {
        "learning_rate": 0.0005, "learning_rate_schedule": "linear",
        "batch_size": 256, "buffer_size": 1000000, "buffer_init_steps": 20000,
        "tau": 0.005, "steps_per_update": 1.0, "save_replay_buffer": True,
        "init_entcoef": 0.1, "n_quantiles": 25, "n_to_drop": 2,
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "memory": {
                "memory_size": 256,
                "sequence_length": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.99,
                "strength": 1.0,
                "network_settings": {
                    "normalize": True,
                    "hidden_units": 256,
                    "num_layers": 2
                }
            }
        }
    },
    "dcac": {
        "learning_rate": 0.0005, "learning_rate_schedule": "linear",
        "batch_size": 256, "buffer_size": 1000000, "buffer_init_steps": 20000,
        "tau": 0.005, "steps_per_update": 1.0, "save_replay_buffer": True,
        "init_entcoef": 0.1, "destructive_threshold": 0.0,
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "memory": {
                "memory_size": 256,
                "sequence_length": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.99,
                "strength": 1.0,
                "network_settings": {
                    "normalize": True,
                    "hidden_units": 256,
                    "num_layers": 2
                }
            }
        }
    },
    "crossq": {
        "learning_rate": 0.0005, "learning_rate_schedule": "linear",
        "batch_size": 256, "buffer_size": 1000000, "buffer_init_steps": 20000,
        "tau": 0.005, "steps_per_update": 1.0, "save_replay_buffer": True,
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "memory": {
                "memory_size": 256,
                "sequence_length": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.99,
                "strength": 1.0,
                "network_settings": {
                    "normalize": True,
                    "hidden_units": 256,
                    "num_layers": 2
                }
            }
        }
    },
    "ppo_et": {
        "learning_rate": 0.0003, "beta": 0.005, "epsilon": 0.2, "lambd": 0.95,
        "num_epoch": 3, "buffer_size": 10240, "entropy_temperature": 1.0,
        "adaptive_entropy_temperature": True, "target_entropy": None,
        "reward_signals": {
            "extrinsic": {
                "strength": 1.0,
                "gamma": 0.99
            }
        }
    },
    "ppo_ce": {
        "learning_rate": 0.0003, "beta": 0.005, "epsilon": 0.2, "lambd": 0.95,
        "num_epoch": 3, "buffer_size": 10240, "curiosity_strength": 0.01,
        "curiosity_gamma": 0.99, "curiosity_learning_rate": 0.0003,
        "curiosity_hidden_units": 128, "curiosity_num_layers": 2,
        "imagination_horizon": 5, "use_imagination_augmented": True,
        "reward_signals": {
            "extrinsic": {
                "strength": 1.0,
                "gamma": 0.99
            }
        }
    },
    "sac_ae": {
        "learning_rate": 0.0003, "batch_size": 128, "buffer_size": 50000,
        "buffer_init_steps": 1000, "tau": 0.005, "steps_per_update": 1.0,
        "init_entcoef": 0.1, "latent_size": 512, "ae_learning_rate": 0.001,
        "ae_hidden_units": 256, "ae_num_layers": 2, "world_model_learning_rate": 0.0003,
        "world_model_hidden_units": 256, "world_model_num_layers": 2,
        "use_autoencoder": True, "use_world_model": True, "ae_loss_weight": 1.0,
        "world_model_loss_weight": 1.0, "reconstruction_loss_weight": 1.0,
        "reward_signals": {
            "extrinsic": {
                "strength": 1.0,
                "gamma": 0.99
            }
        }
    },
    "drqv2": {
        "learning_rate": 0.0005, "learning_rate_schedule": "linear",
        "batch_size": 256, "buffer_size": 1000000, "buffer_init_steps": 20000,
        "tau": 0.005, "steps_per_update": 1.0, "save_replay_buffer": True, "init_entcoef": 0.1,
        "image_pad": 4
    }
}





class TrainingWorker(QThread):
    output_received = pyqtSignal(str)
    model_exported = pyqtSignal(str, str) # Path, RunID
    finished = pyqtSignal(int)

    def __init__(self, command, env=None, name=None):
        super().__init__()
        self.command = command
        self.env = env or os.environ.copy()
        self.name = name
        self.process = None

    def run(self):
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=self.env,
                shell=False
            )

            if self.process.stdout:
                for line in self.process.stdout:
                    line_str = line.strip()
                    
                    # Detect model export event in real-time
                    if "[INFO] Exported" in line_str and ".onnx" in line_str:
                        try:
                            # Extract path: [INFO] Exported results/path/model.onnx
                            parts = line_str.split("Exported ")
                            if len(parts) > 1:
                                path = parts[1].strip()
                                # We need the run_id which is part of the command or name
                                run_id = ""
                                for i, arg in enumerate(self.command):
                                    if arg == "--run-id" and i+1 < len(self.command):
                                        run_id = self.command[i+1]
                                
                                self.model_exported.emit(path, run_id)
                        except:
                            pass

                    if self.name:
                        line_str = f"[{self.name}] {line_str}"
                    self.output_received.emit(line_str)

            exit_code = self.process.wait()
            self.finished.emit(exit_code)
        except Exception as e:
            msg = f"Error starting process: {str(e)}"
            if self.name:
                msg = f"[{self.name}] {msg}"
            self.output_received.emit(msg)
            self.finished.emit(-1)

    def stop(self):
        if self.process:
            self.process.terminate()


class AlgorithmSettingsPage(QWidget):


    """Dynamically generated settings page for a specific algorithm"""


    def __init__(self, algo_name, on_back, on_next, is_last=False):


        super().__init__()


        self.algo_name = algo_name


        layout = QVBoxLayout(self)


        

        title = QLabel(f"Hyperparameters for {algo_name.upper()}")


        title.setAlignment(Qt.AlignmentFlag.AlignCenter)


        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #64B5F6; padding: 10px;")


        layout.addWidget(title)




        scroll = QScrollArea()


        scroll.setWidgetResizable(True)


        scroll.setStyleSheet("QScrollArea { border: none; }")


        

        content = QWidget()


        form = QFormLayout(content)


        

        # Hyperparameters specific to this algorithm


        hp_group = QGroupBox("Algorithm Specific Hyperparameters")


        hp_form = QFormLayout()


        defaults = ALGO_DEFAULTS.get(algo_name, {})


        

        self.inputs = {}


        def create_inputs(data, layout, prefix=""):
            for key, value in data.items():
                full_key = f"{prefix}{key}"
                if isinstance(value, dict):
                    group = QGroupBox(key.replace('_', ' ').capitalize())
                    group_layout = QFormLayout()
                    create_inputs(value, group_layout, f"{full_key}.")
                    group.setLayout(group_layout)
                    layout.addRow(group)
                elif isinstance(value, bool):
                    inp = QCheckBox()
                    inp.setChecked(value)
                    layout.addRow(f"{key.replace('_', ' ').capitalize()}:", inp)
                    self.inputs[full_key] = inp
                elif isinstance(value, float):
                    inp = QDoubleSpinBox()
                    inp.setRange(0.0, 1000000.0)
                    inp.setDecimals(6)
                    inp.setValue(value)
                    layout.addRow(f"{key.replace('_', ' ').capitalize()}:", inp)
                    self.inputs[full_key] = inp
                elif isinstance(value, int):
                    inp = QSpinBox()
                    inp.setRange(0, 1000000000)
                    inp.setValue(value)
                    layout.addRow(f"{key.replace('_', ' ').capitalize()}:", inp)
                    self.inputs[full_key] = inp
                elif value is None:
                    inp = QLineEdit("null")
                    layout.addRow(f"{key.replace('_', ' ').capitalize()}:", inp)
                    self.inputs[full_key] = inp
                else:
                    inp = QLineEdit(str(value))
                    layout.addRow(f"{key.replace('_', ' ').capitalize()}:", inp)
                    self.inputs[full_key] = inp

        create_inputs(defaults, hp_form)


        

        hp_group.setLayout(hp_form)


        form.addWidget(hp_group)




        scroll.setWidget(content)


        layout.addWidget(scroll)




        btn_layout = QHBoxLayout()


        back_btn = QPushButton("Back")


        back_btn.clicked.connect(on_back)


        btn_layout.addWidget(back_btn)




        next_text = "Start training" if is_last else "Next Algorithm"


        next_btn = QPushButton(next_text)


        next_btn.setObjectName("start_btn")


        next_btn.clicked.connect(on_next)


        btn_layout.addWidget(next_btn)


        

        layout.addLayout(btn_layout)

    def get_config(self) -> Dict[str, Any]:
        res = {
            "trainer_type": self.algo_name,
            "hyperparameters": {},
            "network_settings": {},
            "reward_signals": {}
        }
        
        # Define PPO-only parameters that should be filtered for other algorithms
        ppo_only = ['beta', 'epsilon', 'lambd', 'num_epoch', 'learning_rate_schedule']
        # Note: actually some off-policy algos might use learning_rate_schedule, 
        # but the error was specifically about 'beta'.
        
        for key, widget in self.inputs.items():
            value = None
            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                value = widget.value()
            elif isinstance(widget, QLineEdit):
                text = widget.text()
                if text.lower() == "null" or text.lower() == "none":
                    value = None
                else:
                    try:
                        if "." in text:
                            value = float(text)
                        else:
                            value = int(text)
                    except ValueError:
                        value = text
            
            parts = key.split('.')
            
            # Filtering logic for non-PPO algorithms
            ppo_variants = ["ppo", "poca", "ppo_et", "ppo_ce"]
            if self.algo_name not in ppo_variants:
                if parts[-1] in ['beta', 'epsilon', 'lambd', 'num_epoch']:
                    continue

            if len(parts) == 1:
                res["hyperparameters"][parts[0]] = value
            else:
                d = res
                for part in parts[:-1]:
                    if part not in d:
                        d[part] = {}
                    d = d[part]
                d[parts[-1]] = value
        
        if not res["network_settings"]: del res["network_settings"]
        if not res["reward_signals"]: del res["reward_signals"]
        
        return res


class SettingsWindow(QMainWindow):
    """Initial settings window with all ML-Agents parameters and algorithm selection"""
    log_signal = pyqtSignal(str) # Safe bridge for logging from other threads

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CognitionLearn - Initial Settings")
        self.resize(800, 800)

        # Enable dark mode


        MainWindow.apply_dark_theme(self)




        # Main layout


        central_widget = QWidget()


        self.setCentralWidget(central_widget)


        self.stacked_widget = QStackedWidget(central_widget)


        # --- CONSOLE PAGE ---


        self.console_page = QWidget()


        console_layout = QVBoxLayout(self.console_page)


        self.console_output = QTextEdit()
        self.console_output.setReadOnly(True)
        self.console_output.setStyleSheet("background-color: #1E1E1E; color: #DCDCDC; font-family: Courier;")
        
        # Connect the safe signal after console_output is initialized
        self.log_signal.connect(self.console_output.append)
        
        console_layout.addWidget(self.console_output)
        
        console_btn_layout = QHBoxLayout()
        self.back_to_config_btn = QPushButton("Back to Settings")
        self.back_to_config_btn.clicked.connect(self.go_back_from_console)
        console_btn_layout.addWidget(self.back_to_config_btn)
        
        self.stop_btn = QPushButton("Stop Training")
        self.stop_btn.setEnabled(False)
        console_btn_layout.addWidget(self.stop_btn)
        
        console_layout.addLayout(console_btn_layout)
        self.stacked_widget.addWidget(self.console_page)


        main_layout = QVBoxLayout(central_widget)


        main_layout.addWidget(self.stacked_widget)




        self.algo_settings_pages = []
        self.algo_settings_pages = []
        self.training_workers = []  # Changed from single worker to list
        self.visual_monitor_orchestrator = None  # Orchestrator for visual monitor
        self.current_training_index = 0




        # --- PAGE 1: General Settings ---


        page1 = QWidget()


        page1_layout = QVBoxLayout(page1)


        

        scroll = QScrollArea()


        scroll.setWidgetResizable(True)


        scroll.setStyleSheet("QScrollArea { border: none; }")


        

        content_widget = QWidget()


        form_layout = QVBoxLayout(content_widget)


        

        # 1. Engine Settings


        engine_group = QGroupBox("Engine Settings")


        engine_form = QFormLayout()


        

        self.width_sb = QSpinBox()


        self.width_sb.setRange(1, 4096)


        self.width_sb.setValue(250)


        engine_form.addRow("Width:", self.width_sb)


        

        self.height_sb = QSpinBox()


        self.height_sb.setRange(1, 4096)


        self.height_sb.setValue(250)


        engine_form.addRow("Height:", self.height_sb)


        

        self.quality_combo = QComboBox()


        self.quality_combo.addItems(["0 (Fastest)", "1", "2", "3", "4", "5 (Beautiful)"])


        self.quality_combo.setCurrentIndex(0)


        engine_form.addRow("Quality Level:", self.quality_combo)


        

        
        # Time Scale: Slider 0-100
        time_scale_layout = QHBoxLayout()
        self.time_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_scale_slider.setRange(0, 100) # 0 to 100
        self.time_scale_slider.setValue(20)
        self.time_scale_label = QLabel("20")
        self.time_scale_slider.valueChanged.connect(lambda v: self.time_scale_label.setText(str(v)))
        
        time_scale_layout.addWidget(self.time_scale_slider)
        time_scale_layout.addWidget(self.time_scale_label)
        
        engine_form.addRow("Time Scale:", time_scale_layout)


        

        self.target_fps_sb = QSpinBox()


        self.target_fps_sb.setRange(-1, 300)


        self.target_fps_sb.setValue(60)


        engine_form.addRow("Target Frame Rate:", self.target_fps_sb)


        

        self.capture_fps_sb = QSpinBox()


        self.capture_fps_sb.setRange(0, 300)


        self.capture_fps_sb.setValue(0)


        engine_form.addRow("Capture Frame Rate:", self.capture_fps_sb)


        

        self.no_graphics_cb = QCheckBox("No Graphics")
        self.no_graphics_cb.setChecked(True)
        engine_form.addRow(self.no_graphics_cb)

        self.visual_monitor_cb = QCheckBox("Visual Monitor (Watch Best Model)")
        self.visual_monitor_cb.setChecked(False)
        # Always visible now, no longer tied to no_graphics_cb
        engine_form.addRow(self.visual_monitor_cb)

        # Visual Monitor Mode Selection
        self.visual_monitor_mode_combo = QComboBox()
        self.visual_monitor_mode_combo.addItems(["Shared Port Mode", "ONNX Loading Mode"])
        self.visual_monitor_mode_combo.setEnabled(False)  # Only enabled when visual monitor is checked

        # Connect the checkbox state to enable/disable the mode combo
        self.visual_monitor_cb.stateChanged.connect(
            lambda state: self.visual_monitor_mode_combo.setEnabled(bool(state))
        )

        engine_form.addRow(QLabel("Monitor Mode:"), self.visual_monitor_mode_combo)

        engine_group.setLayout(engine_form)
        form_layout.addWidget(engine_group)




        # 2. Environment Settings


        env_group = QGroupBox("Environment Settings")


        env_form = QFormLayout()


        

        path_layout = QHBoxLayout()


        self.env_path_le = QLineEdit("/Users/victorbarbosa/git/UNITY Web/UnityMoonLander/Mac/App.app")


        path_layout.addWidget(self.env_path_le)


        self.browse_btn = QPushButton("Browse")


        self.browse_btn.clicked.connect(self.browse_env_path)


        path_layout.addWidget(self.browse_btn)


        env_form.addRow("Env Path:", path_layout)


        

        self.base_port_sb = QSpinBox()


        self.base_port_sb.setRange(1024, 65535)


        self.base_port_sb.setValue(5005)


        env_form.addRow("Base Port:", self.base_port_sb)


        

        self.num_envs_sb = QSpinBox()


        self.num_envs_sb.setRange(1, 100)


        self.num_envs_sb.setValue(2)


        env_form.addRow("Num Envs:", self.num_envs_sb)


        

        self.num_areas_sb = QSpinBox()


        self.num_areas_sb.setRange(1, 100)


        self.num_areas_sb.setValue(1)


        env_form.addRow("Num Areas:", self.num_areas_sb)


        

        self.timeout_sb = QSpinBox()


        self.timeout_sb.setRange(1, 3600)


        self.timeout_sb.setValue(60)


        env_form.addRow("Timeout (s):", self.timeout_sb)


        

        self.seed_sb = QSpinBox()


        self.seed_sb.setRange(-1, 1000000)


        self.seed_sb.setValue(-1)


        env_form.addRow("Seed:", self.seed_sb)


        

        self.max_restarts_sb = QSpinBox()


        self.max_restarts_sb.setRange(-1, 1000)


        self.max_restarts_sb.setValue(60)


        env_form.addRow("Max Lifetime Restarts:", self.max_restarts_sb)


        

        env_group.setLayout(env_form)


        form_layout.addWidget(env_group)




        # 3. Global Trainer Settings


        trainer_group = QGroupBox("Global Trainer Settings")


        trainer_form = QFormLayout()


        

        self.max_steps_sb = QSpinBox()


        self.max_steps_sb.setRange(1, 2000000000)


        self.max_steps_sb.setValue(200000000)


        trainer_form.addRow("Max Steps:", self.max_steps_sb)


        

        self.time_horizon_sb = QSpinBox()


        self.time_horizon_sb.setRange(1, 10000)


        self.time_horizon_sb.setValue(512)


        trainer_form.addRow("Time Horizon:", self.time_horizon_sb)


        

        self.summary_freq_sb = QSpinBox()
        self.summary_freq_sb.setRange(1, 1000000)
        self.summary_freq_sb.setValue(1000)
        trainer_form.addRow("Summary Freq:", self.summary_freq_sb)

        self.checkpoint_interval_sb = QSpinBox()
        self.checkpoint_interval_sb.setRange(1, 1000000)
        self.checkpoint_interval_sb.setValue(5000)
        trainer_form.addRow("Checkpoint Interval:", self.checkpoint_interval_sb)

        self.keep_checkpoints_sb = QSpinBox()
        self.keep_checkpoints_sb.setRange(1, 1000)
        self.keep_checkpoints_sb.setValue(50)
        trainer_form.addRow("Keep Checkpoints:", self.keep_checkpoints_sb)


        

        trainer_group.setLayout(trainer_form)


        form_layout.addWidget(trainer_group)








        # 5. Checkpoint Settings
        checkpoint_group = QGroupBox("Checkpoint Settings")
        checkpoint_form = QFormLayout()
        
        self.behavior_name_le = QLineEdit("MoonlanderAgent")
        checkpoint_form.addRow("Behavior Name:", self.behavior_name_le)
        
        self.run_id_le = QLineEdit("full_control_run")


        checkpoint_form.addRow("Run ID:", self.run_id_le)


        

        self.results_dir_le = QLineEdit("results")


        checkpoint_form.addRow("Results Dir:", self.results_dir_le)


        

        self.train_model_cb = QCheckBox("Train Model")


        self.train_model_cb.setChecked(True)


        checkpoint_form.addRow(self.train_model_cb)


        

        self.resume_cb = QCheckBox("Resume Training")


        checkpoint_form.addRow(self.resume_cb)


        

        self.force_cb = QCheckBox("Force Overwrite")


        checkpoint_form.addRow(self.force_cb)


        

        checkpoint_group.setLayout(checkpoint_form)


        form_layout.addWidget(checkpoint_group)




        # 6. Torch & Debug


        misc_group = QGroupBox("Torch & Misc")


        misc_form = QFormLayout()


        

        self.device_combo = QComboBox()


        self.device_combo.addItems(["cuda", "cpu", "mps"])


        self.device_combo.setCurrentText("cpu")


        misc_form.addRow("Torch Device:", self.device_combo)


        

        self.debug_cb = QCheckBox("Enable Debug Mode")


        misc_form.addRow(self.debug_cb)


        

        misc_group.setLayout(misc_form)


        form_layout.addWidget(misc_group)




        scroll.setWidget(content_widget)


        page1_layout.addWidget(scroll)




        self.next_btn = QPushButton("Next")


        self.next_btn.setObjectName("start_btn")


        self.next_btn.setMinimumHeight(50)


        self.next_btn.clicked.connect(self.go_to_page2)


        page1_layout.addWidget(self.next_btn)




        self.stacked_widget.addWidget(page1)




        # --- PAGE 2: Algorithm Selection ---


        page2 = QWidget()


        page2_layout = QVBoxLayout(page2)


        

        algo_title = QLabel("Algorithm Configuration")


        algo_title.setAlignment(Qt.AlignmentFlag.AlignCenter)


        algo_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #64B5F6; padding: 10px;")


        page2_layout.addWidget(algo_title)




        # Highlight message for environment distribution


        self.remaining_label = QLabel()


        self.remaining_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


        self.remaining_label.setStyleSheet("""

            background-color: #263238; 

            color: #FFD600; 

            font-weight: bold; 

            font-size: 14px; 

            padding: 15px; 

            border-radius: 5px;

            border: 1px solid #FFD600;

            margin-bottom: 10px;

        """)


        self.remaining_label.setWordWrap(True)


        page2_layout.addWidget(self.remaining_label)




        # Mode selection


        mode_group = QGroupBox("Select Training Mode")


        mode_vbox = QVBoxLayout()


        

        self.same_algo_rb = QRadioButton("Same Algorithm")


        self.same_algo_rb.setChecked(True)


        self.same_algo_rb.toggled.connect(self.update_algo_list_ui)


        

        self.diff_algo_rb = QRadioButton("Different Algorithms")


        self.diff_algo_rb.toggled.connect(self.update_algo_list_ui)


        

        mode_vbox.addWidget(self.same_algo_rb)


        mode_vbox.addWidget(self.diff_algo_rb)


        mode_group.setLayout(mode_vbox)


        page2_layout.addWidget(mode_group)




        # Algorithm List Section


        self.algo_list_group = QGroupBox("Available Algorithms")


        self.algo_list_layout = QVBoxLayout()


        self.algo_list_layout.setSpacing(10)


        self.algo_list_group.setLayout(self.algo_list_layout)


        

        # Scroll area for algorithms


        algo_scroll = QScrollArea()


        algo_scroll.setWidgetResizable(True)


        algo_scroll.setMinimumHeight(400)


        algo_scroll.setStyleSheet("QScrollArea { border: 1px solid #505050; border-radius: 5px; }")


        algo_scroll.setWidget(self.algo_list_group)


        page2_layout.addWidget(algo_scroll)




        # Algorithm definitions


        self.algorithms = [


            "ppo", "ppo_et", "ppo_ce", "sac", "sac_ae", 


            "poca", "tdsac", "td3", "tqc", "drqv2", "dcac", "crossq"


        ]


        

        self.algo_radio_widgets = {}


        self.algo_check_widgets = {}


        self.algo_count_widgets = {}




        self.setup_algorithm_widgets()


        

        page2_layout.addStretch()




        btn_layout = QHBoxLayout()


        self.back_btn = QPushButton("Back")


        self.back_btn.setMinimumHeight(40)


        self.back_btn.clicked.connect(self.go_to_page1)


        btn_layout.addWidget(self.back_btn)




        self.finish_btn = QPushButton("Next")


        self.finish_btn.setObjectName("start_btn")


        self.finish_btn.setMinimumHeight(40)


        self.finish_btn.clicked.connect(self.generate_algo_settings_pages)


        btn_layout.addWidget(self.finish_btn)


        

        page2_layout.addLayout(btn_layout)


        

        self.stacked_widget.addWidget(page2)


        

        # Initial UI update


        self.update_algo_list_ui()



    def setup_algorithm_widgets(self):


        """Pre-create both radio and checkbox widgets for the list"""


        # Radio container (Same Algorithm)


        self.radio_container = QWidget()


        radio_layout = QVBoxLayout(self.radio_container)


        for algo in self.algorithms:


            rb = QRadioButton(algo.upper())


            rb.setMinimumHeight(40)


            if algo == "ppo": rb.setChecked(True)


            self.algo_radio_widgets[algo] = rb


            radio_layout.addWidget(rb)


        self.algo_list_layout.addWidget(self.radio_container)




        # Checkbox container (Different Algorithms)


        self.check_container = QWidget()


        check_layout = QVBoxLayout(self.check_container)


        for algo in self.algorithms:


            row_widget = QWidget()


            row_widget.setMinimumHeight(40)


            row = QHBoxLayout(row_widget)


            row.setContentsMargins(0, 0, 0, 0)


            

            cb = QCheckBox(algo.upper())


            sb = QSpinBox()


            sb.setRange(0, 100) # Minimum 0


            sb.setValue(0)      # Default 0


            sb.setEnabled(False)


            

            # Connect events to update the distribution calculation


            cb.toggled.connect(lambda checked, s=sb: s.setEnabled(checked))


            cb.toggled.connect(self.validate_distribution)


            sb.valueChanged.connect(self.validate_distribution)


            

            self.algo_check_widgets[algo] = cb


            self.algo_count_widgets[algo] = sb


            

            row.addWidget(cb)


            row.addStretch()


            row.addWidget(QLabel("Workers:"))


            row.addWidget(sb)


            check_layout.addWidget(row_widget)


        self.algo_list_layout.addWidget(self.check_container)


        # Start at Initial Settings page (index 1)


        self.stacked_widget.setCurrentIndex(1)



    def validate_distribution(self):


        """Calculate remaining environments and update button state"""


        total_available = self.num_envs_sb.value()


        

        if self.same_algo_rb.isChecked():


            self.remaining_label.hide()


            self.finish_btn.setEnabled(True)


            self.finish_btn.setToolTip("")


        else:


            self.remaining_label.show()


            current_sum = 0


            for algo, cb in self.algo_check_widgets.items():


                if cb.isChecked():


                    current_sum += self.algo_count_widgets[algo].value()


            

            remaining = total_available - current_sum


            

            if remaining == 0:


                self.remaining_label.setText("Distribution complete! All environments are allocated.")


                self.remaining_label.setStyleSheet(self.remaining_label.styleSheet().replace("#FFD600", "#81C784")) # Green


                self.finish_btn.setEnabled(True)


            else:


                self.remaining_label.setText(f"Environment Distribution: You have {total_available} environments total. \nRemaining to allocate: {remaining}")


                self.remaining_label.setStyleSheet(self.remaining_label.styleSheet().replace("#81C784", "#FFD600")) # Yellow


                self.finish_btn.setEnabled(False)



    def update_algo_list_ui(self):


        """Switch visibility between radio and checkbox lists"""


        is_same = self.same_algo_rb.isChecked()


        self.radio_container.setVisible(is_same)


        self.check_container.setVisible(not is_same)


        self.validate_distribution()


    def generate_algo_settings_pages(self):


        """Generate configuration pages for each selected algorithm"""


        # Clear previous dynamic pages from stacked widget (index 3 onwards)


        while self.stacked_widget.count() > 3:


            widget = self.stacked_widget.widget(3)


            self.stacked_widget.removeWidget(widget)


            widget.deleteLater()


        

        selected_algos = []


        if self.same_algo_rb.isChecked():


            for name, rb in self.algo_radio_widgets.items():


                if rb.isChecked():


                    selected_algos.append(name)


                    break


        else:


            for name, cb in self.algo_check_widgets.items():


                if cb.isChecked():


                    selected_algos.append(name)


        

        if not selected_algos:


            return




        # Create pages


        for i, algo in enumerate(selected_algos):


            is_last = (i == len(selected_algos) - 1)


            

            # Use current count to determine current index


            def create_callbacks(index):


                def go_back():


                    # Previous page is index + 2 (since first algo page is at index 3)


                    self.stacked_widget.setCurrentIndex(index + 2)


                def go_next():


                    if index == len(selected_algos) - 1:


                        self.go_to_dashboard()


                    else:


                        # Next algo page is at current index + 4


                        self.stacked_widget.setCurrentIndex(index + 4)


                return go_back, go_next


            

            back_cb, next_cb = create_callbacks(i)




            page = AlgorithmSettingsPage(


                algo, 


                on_back=back_cb, 


                on_next=next_cb, 


                is_last=is_last


            )


            self.stacked_widget.addWidget(page)


        

        # Go to first dynamic algo page (now index 3)


        self.stacked_widget.setCurrentIndex(3)



    def go_to_page1(self):


        self.stacked_widget.setCurrentIndex(1)



    def go_to_page2(self):


        self.validate_distribution()


        self.stacked_widget.setCurrentIndex(2)



    def go_to_dashboard(self):
        """Collect configurations and start training directly, potentially with multiple processes"""
        
        # Switch to console page immediately
        self.stacked_widget.setCurrentWidget(self.console_page)
        self.console_output.clear()
        self.back_to_config_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        # Disconnect old stop button connections
        try:
            self.stop_btn.clicked.disconnect()
        except:
            pass
        self.stop_btn.clicked.connect(self.stop_training)
        
        self.training_workers = []
        
        # Common global settings
        global_max_steps = self.max_steps_sb.value()
        global_time_horizon = self.time_horizon_sb.value()
        global_summary_freq = self.summary_freq_sb.value()
        global_checkpoint_interval = self.checkpoint_interval_sb.value()
        global_keep_checkpoints = self.keep_checkpoints_sb.value()
        
        # Helper to prepare config for an algorithm
        def prepare_config_for_algo(page, behavior_name=None):
            config = page.get_config()
            config["max_steps"] = global_max_steps
            config["time_horizon"] = global_time_horizon
            config["summary_freq"] = global_summary_freq
            config["keep_checkpoints"] = global_keep_checkpoints
            config["checkpoint_interval"] = global_checkpoint_interval # Behavior level
            
            # Also add to hyperparameters for redundancy
            if "hyperparameters" not in config:
                config["hyperparameters"] = {}
            config["hyperparameters"]["checkpoint_interval"] = global_checkpoint_interval
            
            # Clean up PPO-specifics if not PPO
            ppo_variants = ["ppo", "poca", "ppo_et", "ppo_ce"]
            if config.get("trainer_type") not in ppo_variants:
                hparams = config.get("hyperparameters", {})
                fields_to_remove = ["beta", "epsilon", "lambd", "num_epoch"]
                for f in fields_to_remove:
                    hparams.pop(f, None)
                config.pop("shared_critic", None)
            
            # Use provided behavior name or algorithm name
            b_name = behavior_name if behavior_name else page.algo_name
            
            return {
                "default_settings": config,
                "behaviors": {
                    b_name: config
                }
            }

        # --- Execution Logic ---
        
        # Capture common arguments from GUI widgets
        common_args = []
        common_args.extend(["--width", str(self.width_sb.value())])
        common_args.extend(["--height", str(self.height_sb.value())])
        common_args.extend(["--quality-level", str(self.quality_combo.currentIndex())])
        common_args.extend(["--time-scale", str(self.time_scale_slider.value())])
        common_args.extend(["--target-frame-rate", str(self.target_fps_sb.value())])
        common_args.extend(["--capture-frame-rate", str(self.capture_fps_sb.value())])
        common_args.extend(["--num-areas", str(self.num_areas_sb.value())])
        common_args.extend(["--timeout-wait", str(self.timeout_sb.value())])
        
        if self.seed_sb.value() != -1:
            common_args.extend(["--seed", str(self.seed_sb.value())])
            
        if self.max_restarts_sb.value() != -1:
            common_args.extend(["--max-lifetime-restarts", str(self.max_restarts_sb.value())])
            
        if self.no_graphics_cb.isChecked():
            # Training workers should ALWAYS use --no-graphics when headless mode is on
            # Visual Monitor is managed separately by orchestrator
            common_args.append("--no-graphics")

        workers_to_launch = [] # Tuple of (cmd_list, name)
        port_map = {} # Map run_id to port

        if self.same_algo_rb.isChecked():
            # SAME ALGORITHM MODE (Single Process)
            selected_pages = []
            for i in range(0, self.stacked_widget.count()):
                page = self.stacked_widget.widget(i)
                if isinstance(page, AlgorithmSettingsPage):
                    selected_pages.append(page)
            
            if not selected_pages:
                self.console_output.append("Error: No algorithms configured.")
                self.back_to_config_btn.setEnabled(True)
                return

            page = selected_pages[0]
            behavior_name_from_gui = self.behavior_name_le.text().strip()
            if not behavior_name_from_gui:
                behavior_name_from_gui = "MoonlanderAgent"
            
            full_config = prepare_config_for_algo(page, behavior_name_from_gui)
            
            run_id = self.run_id_le.text()
            port = self.base_port_sb.value()
            port_map[run_id] = port

            # Save config
            tmp_dir = tempfile.gettempdir()
            config_path = os.path.join(tmp_dir, f"config_{run_id}.yaml")
            try:
                with open(config_path, "w") as f:
                    yaml.dump(full_config, f, sort_keys=False)
            except Exception as e:
                self.console_output.append(f"Error saving config: {str(e)}")
                self.back_to_config_btn.setEnabled(True)
                return

            # Construct Command
            cmd = [
                sys.executable, "-m", "mlagents.trainers.learn",
                config_path,
                "--run-id", run_id
            ]
            cmd.extend(common_args)
            
            if self.env_path_le.text():
                cmd.extend(["--env", self.env_path_le.text()])
                actual_num_envs = self.num_envs_sb.value()
                cmd.extend(["--num-envs", str(actual_num_envs)])
                cmd.extend(["--base-port", str(port)])
            
            if self.resume_cb.isChecked():
                cmd.append("--resume")
            if self.force_cb.isChecked():
                cmd.append("--force")
                
            workers_to_launch.append((cmd, None))

        else:
            # DIFFERENT ALGORITHMS MODE (Multiple Parallel Processes)
            behavior_name_from_gui = self.behavior_name_le.text().strip()
            if not behavior_name_from_gui:
                behavior_name_from_gui = "MoonlanderAgent"

            selected_pages = []
            for i in range(0, self.stacked_widget.count()):
                page = self.stacked_widget.widget(i)
                if isinstance(page, AlgorithmSettingsPage):
                    selected_pages.append(page)
            
            base_port = self.base_port_sb.value()
            
            for i, page in enumerate(selected_pages):
                algo_name = page.algo_name
                if algo_name in self.algo_count_widgets:
                    num_envs = self.algo_count_widgets[algo_name].value()
                else:
                    num_envs = 1
                
                if num_envs <= 0:
                    continue
                
                run_id = f"{self.run_id_le.text()}_{algo_name}"
                assigned_port = base_port + (i * 100)
                port_map[run_id] = assigned_port
                
                full_config = prepare_config_for_algo(page, behavior_name_from_gui)
                
                # Save config
                tmp_dir = tempfile.gettempdir()
                config_path = os.path.join(tmp_dir, f"config_{run_id}.yaml")
                try:
                    with open(config_path, "w") as f:
                        yaml.dump(full_config, f, sort_keys=False)
                except Exception as e:
                    self.console_output.append(f"Error saving config for {algo_name}: {str(e)}")
                    continue
                
                # Construct Command
                cmd = [
                    sys.executable, "-m", "mlagents.trainers.learn",
                    config_path,
                    "--run-id", run_id
                ]
                cmd.extend(common_args)
                
                if self.env_path_le.text():
                    cmd.extend(["--env", self.env_path_le.text()])
                    cmd.extend(["--num-envs", str(num_envs)])
                    cmd.extend(["--base-port", str(assigned_port)])
                
                if self.resume_cb.isChecked():
                    cmd.append("--resume")
                if self.force_cb.isChecked():
                    cmd.append("--force")
                
                workers_to_launch.append((cmd, algo_name.upper()))

        # Launch Visual Monitor Orchestrator FIRST if enabled
        if self.visual_monitor_cb.isChecked():
            self._launch_visual_monitor_orchestrator(port_map)
            # Give dummy a moment to start before training workers
            import time
            time.sleep(3)
        # Launch all workers
        self.console_output.append(f"Launching {len(workers_to_launch)} training process(es)...")
        
        for cmd, name in workers_to_launch:
            self.console_output.append(f"Starting {name if name else 'Training'}...")
            self.console_output.append(f"Command: {' '.join(cmd)}\n")
            
            worker = TrainingWorker(cmd, name=name)
            worker.output_received.connect(self.log_signal.emit)
            
            # Connect model export event to visual monitor orchestrator
            if self.visual_monitor_orchestrator:
                worker.model_exported.connect(self.visual_monitor_orchestrator.handle_model_exported)
            
            worker.finished.connect(lambda code, n=name: self.on_training_finished(code, n))
            
            worker.start()
            self.training_workers.append(worker)



    def _launch_visual_monitor_orchestrator(self, port_map=None):
        """Launch the visual monitor orchestrator in a separate thread"""
        try:
            from mlagents.trainers.visual_monitor_orchestrator import VisualMonitorOrchestrator
            
            # Get results directory from GUI
            results_dir = self.results_dir_le.text().strip() or "results"
            
            # Get engine settings
            engine_settings = {
                "width": self.width_sb.value(),
                "height": self.height_sb.value(),
                "quality_level": self.quality_combo.currentIndex(),
                "results_dir": results_dir
            }
            
            # Use a port different from training workers
            monitor_port = self.base_port_sb.value() + 10000
            
            # Get behavior name from GUI
            behavior_name = self.behavior_name_le.text().strip()
            if not behavior_name:
                behavior_name = "MoonlanderAgent"

            self.console_output.append("\n[Visual Monitor] Starting orchestrator (dummy environment)...")
            self.console_output.append(f"[Visual Monitor] Monitor port: {monitor_port}")
            self.console_output.append("[Visual Monitor] Will automatically load the best champion ONNX model.\n")
            
            # Determine visual monitor mode
            use_onnx_loading = (self.visual_monitor_mode_combo.currentText() == "ONNX Loading Mode")

            # Create orchestrator
            self.visual_monitor_orchestrator = VisualMonitorOrchestrator(
                results_dir=results_dir,
                env_path=self.env_path_le.text(),
                base_port=monitor_port,
                engine_settings=engine_settings,
                update_interval=50000,  # Check for updates every 50k steps
                behavior_name=behavior_name,
                log_callback=self.log_signal.emit,
                port_map=port_map,
                use_onnx_loading=use_onnx_loading
            )
            
            # Start in background thread
            self.visual_monitor_orchestrator.start()
            
            self.console_output.append("[Visual Monitor] Orchestrator started!")
            self.console_output.append("[Visual Monitor] Check for Unity window opening...")
            
        except ImportError as e:
            self.console_output.append(f"[Visual Monitor] ERROR: Could not import orchestrator module: {e}")
            logger.error(f"Import error for visual monitor: {e}", exc_info=True)
        except Exception as e:
            self.console_output.append(f"[Visual Monitor] ERROR: Failed to start orchestrator: {e}")
            logger.error(f"Failed to start visual monitor orchestrator: {e}", exc_info=True)

    def stop_training(self):
        if self.visual_monitor_orchestrator:
            self.console_output.append("\nStopping visual monitor orchestrator...")
            try:
                self.visual_monitor_orchestrator.stop()
                self.visual_monitor_orchestrator = None
            except Exception as e:
                logger.error(f"Error stopping orchestrator: {e}")
        
        if self.training_workers:
            self.console_output.append("\nStopping all training processes...")
            for worker in self.training_workers:
                if worker.isRunning():
                    worker.stop()
            self.stop_btn.setEnabled(False)
            # We don't immediately enable back button, wait for signals

    def on_training_finished(self, exit_code, name=None):
        msg = f"\nTraining process {'['+name+']' if name else ''} finished with exit code {exit_code}"
        self.console_output.append(msg)
        
        # Check if all workers are done
        all_done = True
        for worker in self.training_workers:
            if worker.isRunning():
                all_done = False
                break
        
        if all_done:
            self.console_output.append("\nAll training processes finished.")
            self.stop_btn.setEnabled(False)
            self.back_to_config_btn.setEnabled(True)

    def go_back_from_console(self):
        # Go back to the last algorithm settings page
        self.stacked_widget.setCurrentIndex(self.stacked_widget.count() - 1)



    def browse_env_path(self):


        file_path, _ = QFileDialog.getOpenFileName(self, "Select Unity Environment")


        if file_path:


            self.env_path_le.setText(file_path)




class MainWindow(QMainWindow):


    """Main application window"""




    def __init__(self):


        super().__init__()


        self.setWindowTitle("CognitionLearn - Advanced ML-Agents Interface")


        self.setGeometry(100, 100, 800, 600)




        # Enable dark mode


        self.set_dark_theme()




        # Set application font


        font = QFont("Arial", 10)


        self.setFont(font)




        # Create central widget and layout


        central_widget = QWidget()


        self.setCentralWidget(central_widget)


        main_layout = QVBoxLayout(central_widget)




        # Title label


        title_label = QLabel("CognitionLearn - Advanced ML-Agents Interface")


        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)


        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #64B5F6; padding: 10px;")


        main_layout.addWidget(title_label)


        

        # Tabs


        tab_widget = QTabWidget()


        

        # Config tab


        self.config_tab = TrainingConfigWidget()


        tab_widget.addTab(self.config_tab, "Configuration")


        

        # Control tab


        self.control_tab = TrainingControlWidget()


        tab_widget.addTab(self.control_tab, "Training Control")


        

        # Console output tab


        self.console_tab = QWidget()


        console_layout = QVBoxLayout()


        self.console_output = QTextEdit()


        self.console_output.setReadOnly(True)


        self.console_output.setStyleSheet("""

            background-color: #1E1E1E;

            color: #DCDCDC;

            border: 1px solid #505050;

            border-radius: 3px;

            padding: 5px;

        """)


        console_layout.addWidget(self.console_output)


        self.console_tab.setLayout(console_layout)


        tab_widget.addTab(self.console_tab, "Console Output")


        

        main_layout.addWidget(tab_widget)


        

        # Status bar


        self.status_bar = self.statusBar()


        self.status_bar.showMessage("Ready")


        

        # Initialize console with welcome message


        self.console_output.append("Welcome to CognitionLearn!")


        self.console_output.append("Advanced interface for ML-Agents training.")


        self.console_output.append("Use this GUI to configure and control your training sessions.")


        self.console_output.append("")



    @staticmethod


    def apply_dark_theme(window):


        """Configure dark theme for any window"""


        app = QApplication.instance()


        if app is not None:


            # Create a dark color scheme


            dark_palette = app.style().standardPalette()


            dark_palette.setColor(dark_palette.ColorRole.Window, QColor(30, 30, 30))


            dark_palette.setColor(dark_palette.ColorRole.WindowText, QColor(220, 220, 220))


            dark_palette.setColor(dark_palette.ColorRole.Base, QColor(45, 45, 45))


            dark_palette.setColor(dark_palette.ColorRole.AlternateBase, QColor(60, 60, 60))


            dark_palette.setColor(dark_palette.ColorRole.ToolTipBase, QColor(30, 30, 30))


            dark_palette.setColor(dark_palette.ColorRole.ToolTipText, QColor(220, 220, 220))


            dark_palette.setColor(dark_palette.ColorRole.Text, QColor(220, 220, 220))


            dark_palette.setColor(dark_palette.ColorRole.Button, QColor(50, 50, 50))


            dark_palette.setColor(dark_palette.ColorRole.ButtonText, QColor(220, 220, 220))


            dark_palette.setColor(dark_palette.ColorRole.BrightText, QColor(240, 240, 240))


            dark_palette.setColor(dark_palette.ColorRole.Highlight, QColor(61, 125, 189))


            dark_palette.setColor(dark_palette.ColorRole.HighlightedText, QColor(0, 0, 0))




            app.setPalette(dark_palette)




            # Set application style sheet for dark theme


            app.setStyleSheet("""

                QMainWindow, QWidget, QTabWidget, QGroupBox {

                    background-color: #1E1E1E;

                    color: #DCDCDC;

                }

                QLabel {

                    color: #DCDCDC;

                }

                QPushButton {

                    background-color: #323232;

                    color: #DCDCDC;

                    border: 1px solid #505050;

                    padding: 5px;

                    border-radius: 3px;

                }

                QPushButton:hover {

                    background-color: #3C3C3C;

                    border: 1px solid #606060;

                }

                QPushButton:pressed {

                    background-color: #2A2A2A;

                }

                QPushButton#start_btn {

                    background-color: #2E7D32;

                    color: white;

                    font-weight: bold;

                }

                QPushButton#start_btn:hover {

                    background-color: #388E3C;

                }

                QPushButton#start_btn:disabled {

                    background-color: #424242;

                    color: #757575;

                    border: 1px solid #333333;

                }

                QPushButton#pause_btn {

                    background-color: #FF8F00;

                    color: black;

                    font-weight: bold;

                }

                QPushButton#pause_btn:hover {

                    background-color: #FFA000;

                }

                QPushButton#stop_btn {

                    background-color: #C62828;

                    color: white;

                    font-weight: bold;

                }

                QPushButton#stop_btn:hover {

                    background-color: #D32F2F;

                }

                QTextEdit {

                    background-color: #2D2D2D;

                    color: #DCDCDC;

                    border: 1px solid #505050;

                    border-radius: 3px;

                }

                QGroupBox {

                    font-weight: bold;

                    border: 1px solid #505050;

                    border-radius: 5px;

                    margin-top: 1ex;

                    padding-top: 10px;

                }

                QGroupBox::title {

                    subcontrol-origin: margin;

                    left: 10px;

                    padding: 0 5px 0 5px;

                    color: #64B5F6;

                }

                QTabWidget::pane {

                    border: 1px solid #505050;

                    border-radius: 5px;

                }

                QTabBar::tab {

                    background-color: #2D2D2D;

                    color: #A0A0A0;

                    padding: 8px;

                    border: 1px solid #505050;

                    border-bottom-color: #505050;

                    border-top-left-radius: 4px;

                    border-top-right-radius: 4px;

                }

                QTabBar::tab:selected {

                    color: #64B5F6;

                    background-color: #1E1E1E;

                    border-bottom-color: #1E1E1E;

                }

                QStatusBar {

                    background-color: #1E1E1E;

                    color: #DCDCDC;

                }

                QMenuBar {

                    background-color: #2D2D2D;

                    color: #DCDCDC;

                }

                QMenuBar::item {

                    background: transparent;

                }

                QMenuBar::item:selected {

                    background: #3C3C3C;

                }

                QMenuBar::item:pressed {

                    background: #2A2A2A;

                }
            """)



    def set_dark_theme(self):


        """Configure dark theme for the application"""


        MainWindow.apply_dark_theme(self)



def create_logo_pixmap(width=300, height=300):


    """Create a simple logo programmatically"""


    pixmap = QPixmap(width, height)


    pixmap.fill(QColor(30, 30, 30))  # Dark background




    painter = QPainter(pixmap)


    painter.setRenderHint(QPainter.RenderHint.Antialiasing)




    # Set pen for drawing


    pen = QPen(QColor(100, 181, 246))  # Light blue


    pen.setWidth(2)


    painter.setPen(pen)




    # Draw brain-like shape


    brush = QBrush(QColor(74, 111, 165))  # Blue color


    painter.setBrush(brush)




    # Main brain shape (simplified)


    painter.drawEllipse(75, 60, 150, 120)  # Main oval


    painter.drawEllipse(100, 70, 50, 60)   # Left bump


    painter.drawEllipse(150, 70, 50, 60)   # Right bump




    # Neural connections


    painter.setBrush(QBrush(QColor(224, 224, 224)))  # Light color for details


    painter.drawEllipse(120, 100, 15, 15)   # Neural connection 1


    painter.drawEllipse(145, 100, 15, 15)   # Neural connection 2


    painter.drawEllipse(132, 120, 10, 10)   # Neural connection 3




    painter.end()




    return pixmap




# Global references to keep objects alive


main_window_instance = None


splash_instance = None


def launch_gui():
    """Launch the main GUI window"""
    if not GUI_AVAILABLE:


        print("PyQt6 is not available. Please install it using: pip install PyQt6")


        return




    app = QApplication.instance()


    if app is None:


        app = QApplication(sys.argv)


        app.setStyle('Fusion')




    # Try to load the logo image


    # Path: ml-agents/cognitionlearn/ico/logo.png


    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


    logo_path = os.path.join(base_dir, "cognitionlearn", "ico", "logo.png")


    

    if os.path.exists(logo_path):


        logo_pixmap = QPixmap(logo_path)


        if not logo_pixmap.isNull():


            # Scale image to be prominent


            logo_pixmap = logo_pixmap.scaled(800, 500, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


        else:


            logo_pixmap = create_logo_pixmap(400, 400)


    else:


        logo_pixmap = create_logo_pixmap(400, 400)




    # Show splash screen


    global splash_instance


    splash_instance = QSplashScreen(logo_pixmap)


    # Ensure it stays on top and is visible


    splash_instance.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)


    splash_instance.show()




    # Process events to ensure the splash screen is drawn


    app.processEvents()




    # Create main window after a delay (5 seconds for better visibility)


    QTimer.singleShot(5000, lambda: show_main_window(app, splash_instance))




    # If running as standalone app


    if not QApplication.instance().property('is_subapp'):


        sys.exit(app.exec())



# Global references to keep objects alive


main_window_instance = None


settings_window_instance = None


splash_instance = None


def show_main_window(app, splash):
    """Show the settings window and close splash screen"""
    global settings_window_instance


    

    # Create and show settings window (first screen after logo)


    settings_window_instance = SettingsWindow()


    settings_window_instance.show()


    

    # Close splash screen when settings window is ready


    splash.finish(settings_window_instance)




if __name__ == "__main__":


    launch_gui()
