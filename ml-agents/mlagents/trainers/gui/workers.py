import os
import subprocess
from typing import Optional
from PyQt6.QtCore import QThread, pyqtSignal

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
