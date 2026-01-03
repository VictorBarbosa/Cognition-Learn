import argparse
import time
import sys
import os
import glob
import logging
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityWorkerInUseException
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents.trainers.gui.visual_monitor_side_channel import VisualMonitorSideChannel

# Suppress annoying side channel warnings from ml-agents-envs
logging.getLogger("mlagents_envs").setLevel(logging.ERROR)

def get_latest_onnx(results_dir):
    """
    Scans the results directory for the most recently modified .onnx file.
    Returns (algorithm_name, run_id, step, path) or None.
    """
    if not results_dir or not os.path.exists(results_dir):
        return None

    # Pattern: results_dir/RunID/BehaviorName.onnx
    # We want to find the NEWEST onnx file.
    # Note: ML-Agents saves intermediate onnx files as well? Usually final or via checkpoints.
    # Checkpoints are in results_dir/RunID/BehaviorName/checkpoint.pt usually.
    # But for visual loading we prefer ONNX. ML-Agents saves ONNX at the end or periodic?
    # Actually, we might need to look at the checkpoints folder if we want intermediate progress.
    # BUT, converting PT to ONNX at runtime is hard.
    # However, users can configure 'save_model_from_checkpoint' or similar.
    # For this prototype, let's assume we are looking for ANY .onnx file that gets updated/created.
    
    files = glob.glob(os.path.join(results_dir, "**", "*.onnx"), recursive=True)
    if not files:
        return None

    # Get latest modified
    latest_file = max(files, key=os.path.getmtime)
    
    # Try to deduce info from path
    # Expected: .../results/RunID/BehaviorName.onnx
    try:
        path_parts = latest_file.split(os.sep)
        run_id = path_parts[-2] # Folder name
        # Heuristic: Algo name is often part of RunID if we followed our naming convention (Algo_Experiment_Index)
        parts = run_id.split('_')
        algo = parts[0] if len(parts) > 0 else "Unknown"
        
        # We don't know the exact step from just the ONNX filename usually, unless we parse side files.
        # We will use file timestamp as a proxy for "step" or just 0 for now.
        step = int(os.path.getmtime(latest_file)) 
        
        return algo, run_id, step, latest_file
    except:
        return "Unknown", "Unknown", 0, latest_file

def main():
    parser = argparse.ArgumentParser(description="ML-Agents Visual Monitor (Dummy)")
    parser.add_argument("--env", default=None, help="Path to the Unity executable")
    parser.add_argument("--run-id", default="VisualMonitor", help="The run-id for the monitor")
    parser.add_argument("--base-port", type=int, default=5005, help="Base port for connection")
    parser.add_argument("--results-dir", default=None, help="Directory to scan for champion models")
    parser.add_argument("--width", type=int, default=84, help="Screen width")
    parser.add_argument("--height", type=int, default=84, help="Screen height")
    parser.add_argument("--time-scale", type=float, default=20.0, help="Time scale for the environment")
    parser.add_argument("--checkpoint-interval", type=int, default=5000, help="Evaluation interval")
    
    args = parser.parse_args()

    print(f"[VisualMonitor] Starting Dummy Environment...")
    print(f"[VisualMonitor] Run ID: {args.run_id}")
    print(f"[VisualMonitor] Env: {args.env}")
    print(f"[VisualMonitor] Port: {args.base_port}")
    print(f"[VisualMonitor] Scanning: {args.results_dir}")

    # Initialize Side Channels
    engine_config_channel = EngineConfigurationChannel()
    env_params_channel = EnvironmentParametersChannel()
    monitor_channel = VisualMonitorSideChannel()

    try:
        # Initialize Unity Environment in Visual Mode
        env = UnityEnvironment(
            file_name=args.env,
            worker_id=0,
            base_port=args.base_port,
            no_graphics=False,
            side_channels=[engine_config_channel, env_params_channel, monitor_channel]
        )
        
        # Set Engine Configuration
        engine_config_channel.set_configuration_parameters(
            width=args.width,
            height=args.height,
            quality_level=5,
            time_scale=args.time_scale,
            target_frame_rate=60,
            capture_frame_rate=60
        )
        
        print("[VisualMonitor] Environment Connected.")
        env.reset()
        
        last_champion_path = ""
        
        # Phase 2 Loop: Monitor and Update
        while True:
            env.step()
            
            # Simple Orchestrator Logic: Poll every few seconds (non-blocking if we just check time)
            # For this loop, checking every frame is too expensive. Check every 100 frames (~2s at 60fps)
            # Actually env.step() takes time. Let's check periodically.
            
            if args.results_dir and time.time() % 5 < 0.1: # Check roughly every 5 seconds
                print(f"[VisualMonitor] Scanning for champions in: {args.results_dir}")
                champion_info = get_latest_onnx(args.results_dir)
                
                if champion_info:
                    algo, run_id, step, path = champion_info
                    print(f"[VisualMonitor] Found latest: {path} (Step: {step})")
                    
                    if path != last_champion_path:
                        print(f"[VisualMonitor] >>> NEW CHAMPION DETECTED! Sending update to Unity...")
                        print(f"[VisualMonitor] Path: {path}")
                        monitor_channel.send_champion_info(
                            algorithm=algo,
                            port=0, # Unknown source port currently
                            step=step,
                            checkpoint_path=os.path.abspath(path) # Ensure absolute path for Unity
                        )
                        last_champion_path = path
                else:
                    print("[VisualMonitor] No .onnx files found yet.")

    except UnityWorkerInUseException:
        print(f"[VisualMonitor] Error: Port {args.base_port} is already in use.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("[VisualMonitor] Stopping...")
    except Exception as e:
        print(f"[VisualMonitor] Unexpected Error: {e}")
        sys.exit(1)
    finally:
        if 'env' in locals() and env:
            env.close()
        print("[VisualMonitor] Closed.")

if __name__ == "__main__":
    main()
