import sys
import os
import yaml
import tempfile
import time
import socket
from typing import Dict, Any, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel,
    QFileDialog, QComboBox, QGroupBox, QScrollArea, QFormLayout,
    QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QStackedWidget,
    QRadioButton, QSlider, QMessageBox, QMainWindow
)
from PyQt6.QtCore import Qt, pyqtSignal

from mlagents.trainers.gui.config import ALGO_DEFAULTS
from mlagents.trainers.gui.workers import TrainingWorker

class AlgorithmSettingsPage(QWidget):
    """Dynamically generated settings page for a specific algorithm"""
    def __init__(self, algo_name, on_back, on_next, is_last=False, global_resume=False):
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

        # Resume / Initialization Section (New)
        if global_resume:
            self.resume_group = QGroupBox("Resume Configuration")
            self.resume_group.setStyleSheet("QGroupBox { border: 2px solid #FFD600; }")
            res_layout = QHBoxLayout()
            self.checkpoint_le = QLineEdit()
            self.checkpoint_le.setPlaceholderText("Path to .onnx or .pt checkpoint (Leave empty to skip resume for this algo)")
            res_layout.addWidget(self.checkpoint_le)
            self.browse_cp_btn = QPushButton("Browse")
            self.browse_cp_btn.clicked.connect(self.browse_checkpoint)
            res_layout.addWidget(self.browse_cp_btn)
            self.resume_group.setLayout(res_layout)
            form.addRow(self.resume_group)
        else:
            self.checkpoint_le = None

        # Hyperparameters specific to this algorithm
        hp_group = QGroupBox("Algorithm Specific Hyperparameters")
        hp_form = QFormLayout()
        defaults = ALGO_DEFAULTS.get(algo_name, {})

        self.inputs = {}
        def create_inputs(data, layout, prefix=""):
            for key, value in data.items():
                full_key = f"{prefix}{key}"
                if key == "memory":
                    # Special handling for Memory (toggleable section)
                    mem_group = QGroupBox("Memory Settings")
                    mem_group.setCheckable(True)
                    mem_group.setChecked(value is not None)
                    
                    # Container for the actual inputs
                    mem_content = QWidget()
                    mem_layout = QFormLayout(mem_content)
                    
                    # Defaults if None
                    defaults = value if value else {"sequence_length": 64, "memory_size": 128}
                    create_inputs(defaults, mem_layout, f"{full_key}.")
                    
                    # Set group layout
                    group_layout = QVBoxLayout()
                    group_layout.setContentsMargins(0, 5, 0, 0)
                    group_layout.addWidget(mem_content)
                    mem_group.setLayout(group_layout)
                    
                    # Connect toggled to hide/show content
                    mem_group.toggled.connect(mem_content.setVisible)
                    # Initial state
                    mem_content.setVisible(mem_group.isChecked())
                    
                    layout.addRow(mem_group)
                    self.inputs[full_key] = mem_group
                elif isinstance(value, dict):
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
                elif value is None:
                    inp = QLineEdit("null")
                    layout.addRow(f"{key.replace('_', ' ').capitalize()}:", inp)
                    self.inputs[full_key] = inp
                elif "schedule" in key.lower():
                    inp = QComboBox()
                    inp.addItems(["linear", "constant"])
                    inp.setCurrentText(str(value))
                    layout.addRow(f"{key.replace('_', ' ').capitalize()}:", inp)
                    self.inputs[full_key] = inp
                elif "vis_encode_type" in key.lower():
                    inp = QComboBox()
                    inp.addItems(["simple", "nature_cnn", "resnet", "match3", "fully_connected"])
                    inp.setCurrentText(str(value))
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

    def browse_checkpoint(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Checkpoint", "", "ML-Agents Checkpoint (*.onnx *.pt);;All Files (*)"
        )
        if filename:
            self.checkpoint_le.setText(filename)

    def get_config(self) -> Dict[str, Any]:
        res = {
            "trainer_type": self.algo_name,
            "hyperparameters": {},
            "network_settings": {},
            "reward_signals": {}
        }
        
        for key, widget in self.inputs.items():
            value = None
            if isinstance(widget, QGroupBox):
                # For checkable groups like Memory
                if not widget.isChecked():
                    continue # Skip if unchecked (equivalent to None)
                value = {} # Placeholder
            elif isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                value = widget.value()
            elif isinstance(widget, QComboBox):
                value = widget.currentText()
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
            
            if len(parts) == 1:
                res["hyperparameters"][parts[0]] = value
            else:
                d = res
                for part in parts[:-1]:
                    if part not in d:
                        d[part] = {}
                    d = d[part]
                d[parts[-1]] = value
        
        if not res["network_settings"]:
            del res["network_settings"]
        if not res["reward_signals"]:
            del res["reward_signals"]
        
        return res


class SettingsWindow(QMainWindow):
    """Initial settings window with all ML-Agents parameters and algorithm selection"""
    log_signal = pyqtSignal(str) # Safe bridge for logging from other threads

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CognitionLearn - Initial Settings")
        self.resize(800, 800)

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
        self.training_workers = []  # Changed from single worker to list
        self.current_training_index = 0

        # --- PAGE 1: General Settings ---
        page1 = QWidget()
        page1_layout = QVBoxLayout(page1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        content_widget = QWidget()
        form_layout = QVBoxLayout(content_widget)

        # 0. Load Configuration
        config_group = QGroupBox("Load Configuration (YAML)")
        config_layout = QHBoxLayout()
        
        self.config_path_le = QLineEdit()
        self.config_path_le.setPlaceholderText("Select a YAML configuration file to load settings...")
        config_layout.addWidget(self.config_path_le)
        
        self.browse_config_btn = QPushButton("Browse")
        self.browse_config_btn.clicked.connect(self.browse_config_path)
        config_layout.addWidget(self.browse_config_btn)
        
        self.load_config_btn = QPushButton("Load")
        self.load_config_btn.clicked.connect(self.load_config_from_yaml)
        config_layout.addWidget(self.load_config_btn)
        
        config_group.setLayout(config_layout)
        form_layout.addWidget(config_group)

        # 1. Engine Settings
        engine_group = QGroupBox("Engine Settings")
        engine_form = QFormLayout()

        self.width_sb = QSpinBox()
        self.width_sb.setRange(1, 4096)
        self.width_sb.setValue(84)
        engine_form.addRow("Width:", self.width_sb)

        self.height_sb = QSpinBox()
        self.height_sb.setRange(1, 4096)
        self.height_sb.setValue(84)
        engine_form.addRow("Height:", self.height_sb)

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["0 (Fastest)", "1", "2", "3", "4", "5 (Beautiful)"])
        self.quality_combo.setCurrentIndex(5)
        engine_form.addRow("Quality Level:", self.quality_combo)
        
        # Time Scale: Slider 0-100
        time_scale_layout = QHBoxLayout()
        self.time_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_scale_slider.setRange(0, 100)
        self.time_scale_slider.setValue(20)
        self.time_scale_label = QLabel("100")
        self.time_scale_slider.valueChanged.connect(lambda v: self.time_scale_label.setText(str(v)))
        
        time_scale_layout.addWidget(self.time_scale_slider)
        time_scale_layout.addWidget(self.time_scale_label)
        engine_form.addRow("Time Scale:", time_scale_layout)

        self.target_fps_sb = QSpinBox()
        self.target_fps_sb.setRange(-1, 300)
        self.target_fps_sb.setValue(-1)
        engine_form.addRow("Target Frame Rate:", self.target_fps_sb)

        self.capture_fps_sb = QSpinBox()
        self.capture_fps_sb.setRange(0, 300)
        self.capture_fps_sb.setValue(60)
        engine_form.addRow("Capture Frame Rate:", self.capture_fps_sb)

        self.no_graphics_cb = QCheckBox("No Graphics")
        self.no_graphics_cb.setChecked(True)
        engine_form.addRow(self.no_graphics_cb)

        self.no_graphics_monitor_cb = QCheckBox("No Graphics Monitor")
        self.no_graphics_monitor_cb.setChecked(False)
        engine_form.addRow(self.no_graphics_monitor_cb)

        self.visual_monitor_cb = QCheckBox("Visual Monitor (Watch Best Model)")
        self.visual_monitor_cb.setChecked(False)
        engine_form.addRow(self.visual_monitor_cb)

        engine_group.setLayout(engine_form)
        form_layout.addWidget(engine_group)

        # 2. Environment Settings
        env_group = QGroupBox("Environment Settings")
        env_form = QFormLayout()

        path_layout = QHBoxLayout()
        self.env_path_le = QLineEdit("")
        path_layout.addWidget(self.env_path_le)
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_env_path)
        path_layout.addWidget(self.browse_btn)
        env_form.addRow("Env Path:", path_layout)

        self.env_args_le = QLineEdit("")
        env_form.addRow("Env Args:", self.env_args_le)

        self.base_port_sb = QSpinBox()
        self.base_port_sb.setRange(1024, 65535)
        self.base_port_sb.setValue(5005)
        env_form.addRow("Base Port:", self.base_port_sb)

        self.num_envs_sb = QSpinBox()
        self.num_envs_sb.setRange(1, 200)
        self.num_envs_sb.setValue(1)
        env_form.addRow("Num Envs:", self.num_envs_sb)

        self.num_areas_sb = QSpinBox()
        self.num_areas_sb.setRange(1, 200)
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
        self.max_restarts_sb.setValue(10)
        env_form.addRow("Max Lifetime Restarts:", self.max_restarts_sb)

        self.restarts_limit_n_sb = QSpinBox()
        self.restarts_limit_n_sb.setValue(1)
        env_form.addRow("Restarts Rate Limit N:", self.restarts_limit_n_sb)

        self.restarts_limit_period_sb = QSpinBox()
        self.restarts_limit_period_sb.setValue(60)
        env_form.addRow("Restarts Rate Limit Period (s):", self.restarts_limit_period_sb)

        env_group.setLayout(env_form)
        form_layout.addWidget(env_group)

        # 3. Global Trainer Settings
        trainer_group = QGroupBox("Global Trainer Settings")
        trainer_form = QFormLayout()

        self.max_steps_sb = QSpinBox()
        self.max_steps_sb.setRange(1, 2000000000)
        self.max_steps_sb.setValue(500000)
        trainer_form.addRow("Max Steps:", self.max_steps_sb)

        self.time_horizon_sb = QSpinBox()
        self.time_horizon_sb.setRange(1, 10000)
        self.time_horizon_sb.setValue(64)
        trainer_form.addRow("Time Horizon:", self.time_horizon_sb)

        self.summary_freq_sb = QSpinBox()
        self.summary_freq_sb.setRange(1, 1000000)
        self.summary_freq_sb.setValue(5000)
        trainer_form.addRow("Summary Freq:", self.summary_freq_sb)

        self.checkpoint_interval_sb = QSpinBox()
        self.checkpoint_interval_sb.setRange(1, 1000000)
        self.checkpoint_interval_sb.setValue(5000)
        trainer_form.addRow("Checkpoint Interval:", self.checkpoint_interval_sb)

        self.keep_checkpoints_sb = QSpinBox()
        self.keep_checkpoints_sb.setRange(1, 1000)
        self.keep_checkpoints_sb.setValue(5)
        trainer_form.addRow("Keep Checkpoints:", self.keep_checkpoints_sb)

        trainer_group.setLayout(trainer_form)
        form_layout.addWidget(trainer_group)

        # 4. Environment Parameters (Dictionary)
        param_group = QGroupBox("Environment Parameters (Key-Value)")
        param_layout = QVBoxLayout()
        self.params_table = QWidget()
        self.params_table_layout = QFormLayout(self.params_table)
        param_layout.addWidget(self.params_table)
        
        add_param_btn = QPushButton("Add Parameter")
        add_param_btn.clicked.connect(self.add_env_parameter)
        param_layout.addWidget(add_param_btn)
        
        param_group.setLayout(param_layout)
        form_layout.addWidget(param_group)

        # 4. Checkpoint Settings
        checkpoint_group = QGroupBox("Checkpoint Settings")
        checkpoint_layout = QVBoxLayout()
        
        # Row 1: ID and Paths
        cp_grid = QFormLayout()
        
        self.behavior_name_le = QLineEdit("MoonlanderAgent")
        cp_grid.addRow("Behavior Name:", self.behavior_name_le)
        
        self.run_id_le = QLineEdit("ppo")
        cp_grid.addRow("Run ID:", self.run_id_le)

        results_path_layout = QHBoxLayout()
        self.results_dir_le = QLineEdit("results")
        results_path_layout.addWidget(self.results_dir_le)
        self.browse_results_btn = QPushButton("Browse")
        self.browse_results_btn.clicked.connect(self.browse_results_path)
        results_path_layout.addWidget(self.browse_results_btn)
        cp_grid.addRow("Results Dir:", results_path_layout)
        
        self.init_from_le = QLineEdit("")
        cp_grid.addRow("Initialize From:", self.init_from_le)
        checkpoint_layout.addLayout(cp_grid)

        # Row 2: Radio Groups
        flags_layout = QHBoxLayout()
        
        # Group A: Execution Mode
        mode_box = QGroupBox("Mode")
        mode_vbox = QVBoxLayout()
        self.train_model_rb = QRadioButton("Train Model")
        self.train_model_rb.setChecked(True)
        self.inference_rb = QRadioButton("Inference")
        mode_vbox.addWidget(self.train_model_rb)
        mode_vbox.addWidget(self.inference_rb)
        mode_box.setLayout(mode_vbox)
        flags_layout.addWidget(mode_box)

        # Group B: Run State
        state_box = QGroupBox("Run State")
        state_vbox = QVBoxLayout()
        self.new_run_rb = QRadioButton("New Run (Default)")
        self.new_run_rb.setChecked(True)
        self.resume_rb = QRadioButton("Resume")
        self.force_rb = QRadioButton("Force (Overwrite)")
        state_vbox.addWidget(self.new_run_rb)
        state_vbox.addWidget(self.resume_rb)
        state_vbox.addWidget(self.force_rb)
        state_box.setLayout(state_vbox)
        flags_layout.addWidget(state_box)

        checkpoint_layout.addLayout(flags_layout)
        checkpoint_group.setLayout(checkpoint_layout)
        form_layout.addWidget(checkpoint_group)

        # 5. Torch & Debug
        misc_group = QGroupBox("Torch & Misc")
        misc_form = QFormLayout()

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        self.device_combo.setCurrentText("cpu")
        misc_form.addRow("Torch Device:", self.device_combo)

        self.debug_cb = QCheckBox("Enable Debug Mode")
        self.debug_cb.setChecked(False)
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
        self.algo_list_layout.setSpacing(2) # Tight spacing
        self.algo_list_layout.setContentsMargins(5, 5, 5, 5)
        self.algo_list_group.setLayout(self.algo_list_layout)

        # Scroll area for algorithms
        algo_scroll = QScrollArea()
        algo_scroll.setWidgetResizable(True)
        algo_scroll.setMinimumHeight(150)
        algo_scroll.setStyleSheet("QScrollArea { border: 1px solid #505050; border-radius: 5px; }")
        algo_scroll.setWidget(self.algo_list_group)
        page2_layout.addWidget(algo_scroll)

        # Algorithm definitions
        self.algorithms = [
            "ppo",
            "ppo_et", 
            "ppo_ce",
            "sac",
            # "sac_ae",
            # "poca",
            "tdsac",
            "td3", 
            "tqc", 
            "drqv2", 
            "dcac", 
            "crossq",
            "dreamer"
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

    def browse_config_path(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select YAML Config", "", "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if filename:
            self.config_path_le.setText(filename)

    def load_config_from_yaml(self):
        path = self.config_path_le.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Error", "Invalid configuration file path.")
            return
            
        try:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
                
            if not config:
                return

            # Basic population of fields if keys exist
            # Engine Settings
            if "engine_settings" in config:
                es = config["engine_settings"]
                if "width" in es: self.width_sb.setValue(es["width"])
                if "height" in es: self.height_sb.setValue(es["height"])
                if "quality_level" in es: self.quality_combo.setCurrentIndex(es["quality_level"])
                if "time_scale" in es: 
                    self.time_scale_slider.setValue(int(es["time_scale"]))
                    self.time_scale_label.setText(str(es["time_scale"]))
                if "target_frame_rate" in es: self.target_fps_sb.setValue(es["target_frame_rate"])
                if "capture_frame_rate" in es: self.capture_fps_sb.setValue(es["capture_frame_rate"])
                if "no_graphics" in es: self.no_graphics_cb.setChecked(es["no_graphics"])
                if "no_graphics_monitor" in es: self.no_graphics_monitor_cb.setChecked(es["no_graphics_monitor"])

            # Environment Settings
            if "env_settings" in config:
                env = config["env_settings"]
                if "env_path" in env and env["env_path"]: self.env_path_le.setText(env["env_path"])
                if "env_args" in env and env["env_args"]: self.env_args_le.setText(" ".join(env["env_args"]))
                if "base_port" in env: self.base_port_sb.setValue(env["base_port"])
                if "num_envs" in env: self.num_envs_sb.setValue(env["num_envs"])
                if "num_areas" in env: self.num_areas_sb.setValue(env["num_areas"])
                if "timeout_wait" in env: self.timeout_sb.setValue(env["timeout_wait"])
                if "seed" in env: self.seed_sb.setValue(env["seed"])
                if "max_lifetime_restarts" in env: self.max_restarts_sb.setValue(env["max_lifetime_restarts"])

            # Checkpoint Settings
            if "checkpoint_settings" in config:
                cp = config["checkpoint_settings"]
                if "run_id" in cp: self.run_id_le.setText(cp["run_id"])
                if "results_dir" in cp: self.results_dir_le.setText(cp["results_dir"])
                if "initialize_from" in cp and cp["initialize_from"]: self.init_from_le.setText(cp["initialize_from"])
                if "resume" in cp: self.resume_rb.setChecked(cp["resume"])
                if "force" in cp: self.force_rb.setChecked(cp["force"])
                if "train_model" in cp: self.train_model_rb.setChecked(cp["train_model"])
                if "inference" in cp: self.inference_rb.setChecked(cp["inference"])

            # Torch Settings
            if "torch_settings" in config:
                ts = config["torch_settings"]
                if "device" in ts: self.device_combo.setCurrentText(ts["device"])

            # Behaviors (Simple assumption: single behavior or just taking first)
            if "behaviors" in config:
                behaviors = config["behaviors"]
                if behaviors:
                    b_name = list(behaviors.keys())[0]
                    self.behavior_name_le.setText(b_name)
                    
            QMessageBox.information(self, "Success", "Configuration loaded successfully (partial).")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load config: {str(e)}")

    def browse_results_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Results Directory", self.results_dir_le.text())
        if dir_path:
            self.results_dir_le.setText(dir_path)

    def add_env_parameter(self):
        # Container for the row
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        
        key_le = QLineEdit()
        key_le.setPlaceholderText("Key")
        val_le = QLineEdit()
        val_le.setPlaceholderText("Value")
        
        remove_btn = QPushButton("X")
        remove_btn.setFixedSize(30, 30)
        # Style it red
        remove_btn.setStyleSheet("""
            QPushButton {
                background-color: #E57373; 
                color: white; 
                font-weight: bold;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #EF5350;
            }
        """)
        
        row_layout.addWidget(key_le)
        row_layout.addWidget(val_le)
        row_layout.addWidget(remove_btn)
        
        self.params_table_layout.addRow(row_widget)
        
        if not hasattr(self, 'param_widgets'):
            self.param_widgets = []
            
        # Store reference to remove later
        entry = {'widget': row_widget, 'key': key_le, 'val': val_le}
        self.param_widgets.append(entry)
        
        # Connect remove button
        remove_btn.clicked.connect(lambda: self.remove_env_parameter(entry))

    def remove_env_parameter(self, entry):
        if entry in self.param_widgets:
            self.params_table_layout.removeRow(entry['widget'])
            self.param_widgets.remove(entry)

    def browse_env_path(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Unity Environment", "", "All Files (*)"
        )
        if filename:
            self.env_path_le.setText(filename)

    def setup_algorithm_widgets(self):
        """Pre-create both radio and checkbox widgets for the list"""
        # Radio container (Same Algorithm)
        self.radio_container = QWidget()
        radio_layout = QVBoxLayout(self.radio_container)
        radio_layout.setSpacing(2) # Tight spacing
        radio_layout.setContentsMargins(5, 5, 5, 5)
        for algo in self.algorithms:
            rb = QRadioButton(algo.upper())
            rb.setMinimumHeight(25) # Reduced height
            if algo == "ppo": rb.setChecked(True)
            self.algo_radio_widgets[algo] = rb
            radio_layout.addWidget(rb)
        radio_layout.addStretch() # Push items to the top
        self.algo_list_layout.addWidget(self.radio_container)

        # Checkbox container (Different Algorithms)
        self.check_container = QWidget()
        check_layout = QVBoxLayout(self.check_container)
        check_layout.setSpacing(2) # Tight spacing
        check_layout.setContentsMargins(5, 5, 5, 5)
        for algo in self.algorithms:
            row_widget = QWidget()
            row_widget.setMinimumHeight(30) # Reduced height
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
        check_layout.addStretch() # Push items to the top
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
        is_global_resume = self.resume_rb.isChecked()
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
                is_last=is_last,
                global_resume=is_global_resume
            )
            self.stacked_widget.addWidget(page)
        
        # Go to first dynamic algo page (now index 3)
        self.stacked_widget.setCurrentIndex(3)

    def go_to_page1(self):
        self.stacked_widget.setCurrentIndex(1)

    def go_to_page2(self):
        # Validation
        errors = []
        
        if not self.env_path_le.text().strip():
            errors.append("Environment Path")
            
        if not self.run_id_le.text().strip():
            errors.append("Run ID")
            
        if not self.results_dir_le.text().strip():
            errors.append("Results Directory")
            
        if not self.behavior_name_le.text().strip():
            errors.append("Behavior Name")
            
        if errors:
            msg = "The following fields cannot be empty:\n\n" + "\n".join(f"- {e}" for e in errors)
            QMessageBox.warning(self, "Validation Error", msg)
            return

        self.validate_distribution()
        self.stacked_widget.setCurrentIndex(2)
        
    def go_back_from_console(self):
        """Allow returning to settings if training stops or fails immediately"""
        self.stacked_widget.setCurrentIndex(1)

    def check_port_free(self, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # 0 means success (connected), so not free
            result = sock.connect_ex(('localhost', port))
            return result != 0
        finally:
            sock.close()

    def find_free_port_block(self, start_port, num_needed):
        current_port = start_port
        # Safety limit to prevent infinite loops
        while current_port < 65000:
            all_free = True
            for i in range(num_needed):
                if not self.check_port_free(current_port + i):
                    all_free = False
                    break
            
            if all_free:
                return current_port
            
            # Jump past the occupied block or just increment
            current_port += 1
        return start_port

    def go_to_dashboard(self):
        """Collect configurations and start training directly"""
        
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

        # Automatic Port Handling
        requested_port = self.base_port_sb.value()
        num_envs = self.num_envs_sb.value()
        
        self.console_output.append(f"Checking availability for {num_envs} ports starting at {requested_port}...")
        free_port = self.find_free_port_block(requested_port, num_envs)
        
        if free_port != requested_port:
            self.console_output.append(f"Port {requested_port} (or subsequent) is busy. Switching to {free_port}.")
            self.base_port_sb.setValue(free_port)
        else:
            self.console_output.append(f"Ports available starting at {free_port}.")
        
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
            
            # Clean up PPO-specifics if not PPO
            ppo_variants = ["ppo", "poca", "ppo_et", "ppo_ce"]
            if config.get("trainer_type") not in ppo_variants:
                hparams = config.get("hyperparameters", {})
                fields_to_remove = ["beta", "epsilon", "lambd", "num_epoch"]
                for f in fields_to_remove:
                    hparams.pop(f, None)
            
            # Auto-correct batch_size if smaller than sequence_length when memory is used
            hparams = config.get("hyperparameters", {})
            net_settings = config.get("network_settings", {})
            memory_settings = net_settings.get("memory", {})
            
            # Ensure batch_size is int
            batch_size = int(hparams.get("batch_size", 1024))
            
            if memory_settings:
                seq_len = int(memory_settings.get("sequence_length", 64))
                if batch_size < seq_len:
                    self.console_output.append(f"[WARNING] Batch size ({batch_size}) is smaller than sequence length ({seq_len}) for {page.algo_name}. Auto-adjusting batch_size to {seq_len}.")
                    batch_size = seq_len
                    hparams["batch_size"] = batch_size

            # Use provided behavior name or algorithm name
            b_name = behavior_name if behavior_name else page.algo_name
            
            return config, b_name

        # --- Build RunOptions Dict ---
        run_options = {
            "behaviors": {},
            "env_settings": {
                "env_path": self.env_path_le.text() or None,
                "env_args": self.env_args_le.text().split() if self.env_args_le.text() else None,
                "base_port": self.base_port_sb.value(),
                "num_envs": self.num_envs_sb.value(),
                "num_areas": self.num_areas_sb.value(),
                "timeout_wait": self.timeout_sb.value(),
                "seed": self.seed_sb.value(),
                "max_lifetime_restarts": self.max_restarts_sb.value(),
                "restarts_rate_limit_n": self.restarts_limit_n_sb.value(),
                "restarts_rate_limit_period_s": self.restarts_limit_period_sb.value(),
            },
            "engine_settings": {
                "width": self.width_sb.value(),
                "height": self.height_sb.value(),
                "quality_level": self.quality_combo.currentIndex(),
                "time_scale": float(self.time_scale_slider.value()),
                "target_frame_rate": self.target_fps_sb.value(),
                "capture_frame_rate": self.capture_fps_sb.value(),
                "no_graphics": self.no_graphics_cb.isChecked(),
                "no_graphics_monitor": self.no_graphics_monitor_cb.isChecked(),
            },
            "checkpoint_settings": {
                "run_id": self.run_id_le.text(),
                "results_dir": self.results_dir_le.text(),
                "initialize_from": self.init_from_le.text() or None,
                "resume": self.resume_rb.isChecked(),
                "force": self.force_rb.isChecked(),
                "train_model": self.train_model_rb.isChecked(),
                "inference": self.inference_rb.isChecked(),
            },
            "torch_settings": {
                "device": self.device_combo.currentText(),
            },
            "debug": self.debug_cb.isChecked(),
            "environment_parameters": {}
        }

        # Collect Environment Parameters
        if hasattr(self, 'param_widgets'):
            for p in self.param_widgets:
                key = p['key'].text().strip()
                val = p['val'].text().strip()
                if key:
                    try:
                        # Try to convert to float/int if possible
                        if "." in val:
                            run_options["environment_parameters"][key] = float(val)
                        else:
                            run_options["environment_parameters"][key] = int(val)
                    except:
                        run_options["environment_parameters"][key] = val

        # Collect Behaviors and determine workers to launch
        workers_to_launch_info = [] # List of (algo_name, page, count)
        
        # Mapping algo_name -> Page
        algo_to_page = {}
        for i in range(self.stacked_widget.count()):
            w = self.stacked_widget.widget(i)
            if isinstance(w, AlgorithmSettingsPage):
                algo_to_page[w.algo_name] = w

        if self.same_algo_rb.isChecked():
            # Find the selected radio
            selected_algo = None
            for algo, rb in self.algo_radio_widgets.items():
                if rb.isChecked():
                    selected_algo = algo
                    break
            if selected_algo and selected_algo in algo_to_page:
                workers_to_launch_info.append((selected_algo, algo_to_page[selected_algo], self.num_envs_sb.value()))
        else:
            for algo, cb in self.algo_check_widgets.items():
                if cb.isChecked() and algo in algo_to_page:
                    count = self.algo_count_widgets[algo].value()
                    if count > 0:
                        workers_to_launch_info.append((algo, algo_to_page[algo], count))
        
        if not workers_to_launch_info:
            QMessageBox.critical(self, "Error", "No algorithms configured.")
            self.go_back_from_console()
            return

        base_run_id = self.run_id_le.text()
        start_base_port = self.base_port_sb.value()
        total_launched_count = 0
        self.training_workers = []

        # --- Launch Visual Monitor (Dummy) - MUST BE FIRST ---
        if self.visual_monitor_cb.isChecked():
            monitor_port = start_base_port + total_launched_count
            
            cmd = [
                sys.executable, "-m", "mlagents.trainers.gui.visual_monitor",
                "--run-id", "VisualMonitor",
                "--base-port", str(monitor_port),
                "--width", str(self.width_sb.value()),
                "--height", str(self.height_sb.value()),
                "--time-scale", "1.0", # Forced real-time
                "--results-dir", self.results_dir_le.text()
            ]
            
            env_path = self.env_path_le.text().strip()
            if env_path:
                cmd.extend(["--env", env_path])
            
            self.console_output.append(f"--- Launching Visual Monitor (Dummy) ---")
            self.console_output.append(f"Command: {' '.join(cmd)}")
            
            worker = TrainingWorker(cmd, name="visual_monitor")
            worker.output_received.connect(self.log_signal.emit)
            worker.finished.connect(self.on_worker_finished)
            worker.start()
            self.training_workers.append(worker)
            
            total_launched_count += 1

        for algo_name, page, worker_count in workers_to_launch_info:
            # Prepare config for this algorithm once
            behavior_name = self.behavior_name_le.text().strip() or "MoonlanderAgent"
            config, b_name = prepare_config_for_algo(page, behavior_name)
            
            # Create a shared run_options template for this algo
            run_options = {
                "behaviors": {b_name: config},
                "env_settings": {
                    "env_path": self.env_path_le.text() or None,
                    "env_args": self.env_args_le.text().split() if self.env_args_le.text() else None,
                    "base_port": start_base_port,
                    "num_envs": worker_count if self.same_algo_rb.isChecked() else 1,
                    "num_areas": self.num_areas_sb.value(),
                    "timeout_wait": self.timeout_sb.value(),
                    "seed": self.seed_sb.value(),
                    "max_lifetime_restarts": self.max_restarts_sb.value(),
                    "restarts_rate_limit_n": self.restarts_limit_n_sb.value(),
                    "restarts_rate_limit_period_s": self.restarts_limit_period_sb.value(),
                },
                "engine_settings": {
                    "width": self.width_sb.value(),
                    "height": self.height_sb.value(),
                    "quality_level": self.quality_combo.currentIndex(),
                    "time_scale": float(self.time_scale_slider.value()),
                    "target_frame_rate": self.target_fps_sb.value(),
                    "capture_frame_rate": self.capture_fps_sb.value(),
                    "no_graphics": self.no_graphics_cb.isChecked(),
                    "no_graphics_monitor": self.no_graphics_monitor_cb.isChecked(),
                },
                "checkpoint_settings": {
                    "run_id": base_run_id,
                    "results_dir": self.results_dir_le.text(),
                    "initialize_from": self.init_from_le.text() or None,
                    "resume": self.resume_rb.isChecked(),
                    "force": self.force_rb.isChecked(),
                    "train_model": self.train_model_rb.isChecked(),
                    "inference": self.inference_rb.isChecked(),
                },
                "torch_settings": {
                    "device": self.device_combo.currentText(),
                },
                "debug": self.debug_cb.isChecked(),
                "environment_parameters": {}
            }

            # Collect Environment Parameters
            if hasattr(self, 'param_widgets'):
                for p in self.param_widgets:
                    key = p['key'].text().strip()
                    val = p['val'].text().strip()
                    if key:
                        try:
                            if "." in val: run_options["environment_parameters"][key] = float(val)
                            else: run_options["environment_parameters"][key] = int(val)
                        except: run_options["environment_parameters"][key] = val

            # Save a common config file for this algorithm
            tmp_dir = tempfile.gettempdir()
            config_path = os.path.join(tmp_dir, f"config_{base_run_id}_{algo_name}.yaml")
            try:
                with open(config_path, "w") as f:
                    yaml.dump(run_options, f, sort_keys=False)
            except Exception as e:
                self.console_output.append(f"Error saving config for {algo_name}: {str(e)}")
                continue

            # Determine iterations: 1 per algorithm (grouping workers)
            worker_port = start_base_port + total_launched_count
            
            # Construct Command
            cmd = [
                sys.executable, "-m", "mlagents.trainers.learn",
                config_path
            ]

            # Run ID Logic
            if self.same_algo_rb.isChecked():
                 cmd.extend(["--run-id", base_run_id])
            else:
                 run_id_formatted = f"{algo_name}_{base_run_id}"
                 cmd.extend(["--run-id", run_id_formatted])
            
            cmd.extend(["--num-envs", str(worker_count)])
            cmd.extend(["--base-port", str(worker_port)])

            # Add extra CLI arguments
            env_path = self.env_path_le.text().strip()
            if env_path: cmd.extend(["--env", env_path])
            
            env_args = self.env_args_le.text().strip()
            if env_args: cmd.extend(["--env-args"] + env_args.split())

            cmd.extend(["--num-areas", str(self.num_areas_sb.value())])
            cmd.extend(["--timeout-wait", str(self.timeout_sb.value())])
            if self.seed_sb.value() != -1: cmd.extend(["--seed", str(self.seed_sb.value())])
            
            cmd.extend(["--width", str(self.width_sb.value())])
            cmd.extend(["--height", str(self.height_sb.value())])
            cmd.extend(["--quality-level", str(self.quality_combo.currentIndex())])
            cmd.extend(["--time-scale", str(self.time_scale_slider.value())])
            cmd.extend(["--target-frame-rate", str(self.target_fps_sb.value())])
            cmd.extend(["--capture-frame-rate", str(self.capture_fps_sb.value())])
            
            if self.no_graphics_cb.isChecked(): cmd.append("--no-graphics")
            
            cmd.extend(["--torch-device", self.device_combo.currentText()])
            cmd.extend(["--results-dir", self.results_dir_le.text()])
            
            # --- Per-Algorithm Resume Logic ---
            algo_checkpoint = ""
            if hasattr(page, 'checkpoint_le') and page.checkpoint_le:
                algo_checkpoint = page.checkpoint_le.text().strip()
            
            is_resuming_this_algo = False
            if self.resume_rb.isChecked():
                if algo_checkpoint:
                    # Global resume is ON and we have a specific checkpoint
                    cmd.append("--resume")
                    cmd.extend(["--initialize-from", algo_checkpoint])
                    is_resuming_this_algo = True
                else:
                    # Global resume is ON but NO checkpoint provided for this algo
                    # User said: "se esse campo estiver vazio então o resume é ignorado para esse algorimo"
                    self.console_output.append(f"[INFO] Skipping resume for {algo_name} because no checkpoint was specified.")
                    # (We don't add --resume)
            
            init_from = self.init_from_le.text().strip()
            if init_from and not is_resuming_this_algo: # Only use global if not specific
                cmd.extend(["--initialize-from", init_from])

            if self.debug_cb.isChecked(): cmd.append("--debug")
            # Add resume/force/inference only if they weren't handled by the specific logic above
            if self.resume_rb.isChecked() and is_resuming_this_algo and "--resume" not in cmd:
                 cmd.append("--resume")
            
            if self.force_rb.isChecked(): cmd.append("--force")
            if self.inference_rb.isChecked(): cmd.append("--inference")
            # Log and Start Process
            name_log = base_run_id if self.same_algo_rb.isChecked() else f"{algo_name}_{base_run_id}"
            self.console_output.append(f"--- Launching Algorithm Group: {algo_name} (Envs: {worker_count}) ---")
            self.console_output.append(f"Command: {' '.join(cmd)}")
            
            worker = TrainingWorker(cmd, name=name_log)
            worker.output_received.connect(self.log_signal.emit)
            worker.finished.connect(self.on_worker_finished)
            worker.start()
            self.training_workers.append(worker)
            
            total_launched_count += worker_count 

        self.console_output.append(f"Total processes launched: {len(self.training_workers)}")
        # Update UI base port for next manual run to avoid collision
        self.base_port_sb.setValue(start_base_port + total_launched_count)

    def on_worker_finished(self, code):
        self.console_output.append(f"Worker finished with code {code}")
        # Check if all finished
        all_finished = True
        for w in self.training_workers:
            if w.isRunning():
                all_finished = False
                break
        
        if all_finished:
            self.stop_btn.setEnabled(False)
            self.back_to_config_btn.setEnabled(True)
            self.console_output.append("All training processes finished.")

    def stop_training(self):
        self.console_output.append("Stopping all workers...")
        for w in self.training_workers:
            if w.isRunning():
                w.stop()