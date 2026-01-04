# Default Hyperparameters for each algorithm

ALGO_DEFAULTS = {
    "ppo": {
        "batch_size": 1024,
        "buffer_size": 6144,
        "learning_rate": 0.00035,
        "beta": 0.005,
        "epsilon": 0.3,
        "lambd": 0.95,
        "num_epoch": 5,
        "shared_critic": False,
        "learning_rate_schedule": "linear",
        "beta_schedule": "linear",
        "epsilon_schedule": "linear",
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "memory": {
                "sequence_length": 64,
                "memory_size": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.95,
                "strength": 0.99
            }
        }
    },
    "sac": {
        "batch_size": 1024,
        "buffer_size": 6144,
        "learning_rate": 0.00035,
        "learning_rate_schedule": "linear",
        "buffer_init_steps": 0,
        "tau": 0.005,
        "steps_per_update": 1.0,
        "save_replay_buffer": False,
        "init_entcoef": 1.0,
        "reward_signal_steps_per_update": 4.0,
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "memory": {
                "sequence_length": 64,
                "memory_size": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.95,
                "strength": 0.99
            }
        }
    },
    "td3": {
        "batch_size": 1024,
        "buffer_size": 6144,
        "learning_rate": 0.00035,
        "learning_rate_schedule": "linear",
        "buffer_init_steps": 0,
        "tau": 0.005,
        "steps_per_update": 1.0,
        "policy_delay": 2,
        "target_noise": 0.2,
        "noise_clip": 0.5,
        "exploration_noise": 0.1,
        "save_replay_buffer": False,
        "reward_signal_steps_per_update": 4.0,
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "memory": {
                "sequence_length": 64,
                "memory_size": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.95,
                "strength": 0.99
            }
        }
    },
    "ppo_et": {
        "batch_size": 1024,
        "buffer_size": 6144,
        "learning_rate": 0.00035,
        "beta": 0.005,
        "epsilon": 0.3,
        "lambd": 0.95,
        "num_epoch": 5,
        "shared_critic": False,
        "learning_rate_schedule": "linear",
        "beta_schedule": "linear",
        "epsilon_schedule": "linear",
        "entropy_temperature": 1.0,
        "adaptive_entropy_temperature": True,
        "target_entropy": None,
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "memory": {
                "sequence_length": 64,
                "memory_size": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.95,
                "strength": 0.99
            }
        }
    },
    "ppo_ce": {
        "batch_size": 1024,
        "buffer_size": 6144,
        "learning_rate": 0.00035,
        "beta": 0.005,
        "epsilon": 0.3,
        "lambd": 0.95,
        "num_epoch": 5,
        "shared_critic": False,
        "learning_rate_schedule": "linear",
        "beta_schedule": "linear",
        "epsilon_schedule": "linear",
        "curiosity_strength": 0.01,
        "curiosity_gamma": 0.99,
        "curiosity_learning_rate": 3e-4,
        "curiosity_hidden_units": 256,
        "curiosity_num_layers": 3,
        "imagination_horizon": 5,
        "use_imagination_augmented": True,
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "memory": {
                "sequence_length": 64,
                "memory_size": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.95,
                "strength": 0.99
            }
        }
    },
    "tdsac": {
        "batch_size": 1024,
        "buffer_size": 6144,
        "learning_rate": 0.00035,
        "learning_rate_schedule": "linear",
        "buffer_init_steps": 0,
        "tau": 0.005,
        "steps_per_update": 1.0,
        "save_replay_buffer": False,
        "init_entcoef": 1.0,
        "reward_signal_steps_per_update": 4.0,
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "memory": {
                "sequence_length": 64,
                "memory_size": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.95,
                "strength": 0.99
            }
        }
    },
    "tqc": {
        "batch_size": 1024,
        "buffer_size": 6144,
        "learning_rate": 0.00035,
        "learning_rate_schedule": "linear",
        "buffer_init_steps": 0,
        "tau": 0.005,
        "steps_per_update": 1.0,
        "save_replay_buffer": False,
        "init_entcoef": 1.0,
        "reward_signal_steps_per_update": 4.0,
        "n_quantiles": 25,
        "n_to_drop": 2,
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "memory": {
                "sequence_length": 64,
                "memory_size": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.95,
                "strength": 0.99
            }
        }
    },
    "drqv2": {
        "batch_size": 1024,
        "buffer_size": 6144,
        "learning_rate": 0.00035,
        "learning_rate_schedule": "linear",
        "buffer_init_steps": 0,
        "tau": 0.005,
        "steps_per_update": 1.0,
        "save_replay_buffer": False,
        "init_entcoef": 1.0,
        "reward_signal_steps_per_update": 4.0,
        "image_pad": 4,
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "memory": {
                "sequence_length": 64,
                "memory_size": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.95,
                "strength": 0.99
            }
        }
    },
    "dcac": {
        "batch_size": 1024,
        "buffer_size": 6144,
        "learning_rate": 0.00035,
        "learning_rate_schedule": "linear",
        "buffer_init_steps": 0,
        "tau": 0.005,
        "steps_per_update": 1.0,
        "save_replay_buffer": False,
        "init_entcoef": 1.0,
        "reward_signal_steps_per_update": 4.0,
        "destructive_threshold": 0.0,
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "memory": {
                "sequence_length": 64,
                "memory_size": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.95,
                "strength": 0.99
            }
        }
    },
    "crossq": {
        "batch_size": 1024,
        "buffer_size": 6144,
        "learning_rate": 0.00035,
        "learning_rate_schedule": "linear",
        "buffer_init_steps": 0,
        "tau": 0.005,
        "steps_per_update": 1.0,
        "save_replay_buffer": False,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "noise_clip": 0.5,
        "reward_signal_steps_per_update": 4.0,
        "network_settings": {
            "normalize": True,
            "hidden_units": 256,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "memory": {
                "sequence_length": 64,
                "memory_size": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.95,
                "strength": 0.99
            }
        }
    },
    "dreamer": {
        "batch_size": 16, # Dreamer uses small batch sizes but processes sequences
        "buffer_size": 100000,
        "learning_rate": 1e-4,
        "learning_rate_schedule": "constant",
        "buffer_init_steps": 1000,
        "steps_per_update": 1.0,
        "save_replay_buffer": True,
        "reward_signal_steps_per_update": 1.0,
        
        # Dreamer Specific
        "batch_length": 64,
        "horizon": 15,
        "hidden_units": 512,
        "gru_units": 512,
        "cnn_depth": 32,
        "mlp_layers": 3,
        "model_lr": 1e-4,
        "actor_lr": 8e-5,
        "value_lr": 8e-5,
        "kl_scale": 1.0,
        "free_nats": 3.0,
        
        "network_settings": {
            "normalize": False, # Dreamer handles normalization internally usually
            "hidden_units": 512,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "memory": { # Required for Dreamer to work with ML-Agents buffer structure
                "sequence_length": 64, 
                "memory_size": 512
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.997, # Dreamer uses long horizons
                "strength": 1.0
            }
        }
    }
}
