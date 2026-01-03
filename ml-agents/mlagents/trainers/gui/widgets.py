from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel,
    QGroupBox, QFileDialog
)

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
