# Default Hyperparameters for each algorithm

ALGO_DEFAULTS = {
    "ppo": {
        "batch_size": 2048,
        "buffer_size": 20480,
        "learning_rate": 0.0003,
        "beta": 0.01,
        "epsilon": 0.2,
        "lambd": 0.95,
        "num_epoch": 3,
        "shared_critic": False,
        "learning_rate_schedule": "linear",
        "beta_schedule": "linear",
        "epsilon_schedule": "linear",
        "network_settings": {
            "normalize": True,
            "hidden_units": 512,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "goal_conditioning_type": "hyper",
            "deterministic": False,
            "memory": {
                "sequence_length": 64,
                "memory_size": 256
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.995,
                "strength": 1.0,
                "network_settings": {
                    "normalize": False,
                    "hidden_units": 128,
                    "num_layers": 2,
                    "vis_encode_type": "simple",
                    "memory": None,
                    "goal_conditioning_type": "hyper",
                    "deterministic": False
                }
            }
        }
    },
    "sac": {
        "batch_size": 2048,
        "buffer_size": 200000,
        "learning_rate": 0.0003,
        "learning_rate_schedule": "constant",
        "buffer_init_steps": 5000,
        "tau": 0.005,
        "steps_per_update": 10.0,
        "save_replay_buffer": True,
        "init_entcoef": 0.1,
        "reward_signal_steps_per_update": 1.0,
        "network_settings": {
            "normalize": True,
            "hidden_units": 512,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "goal_conditioning_type": "hyper",
            "deterministic": False,
            "memory": {
                "sequence_length": 64,
                "memory_size": 256
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.995,
                "strength": 1.0,
                "network_settings": {
                    "normalize": False,
                    "hidden_units": 128,
                    "num_layers": 2,
                    "vis_encode_type": "simple",
                    "memory": None,
                    "goal_conditioning_type": "hyper",
                    "deterministic": False
                }
            }
        }
    },
    "td3": {
        "batch_size": 2048,
        "buffer_size": 200000,
        "learning_rate": 0.0003,
        "learning_rate_schedule": "linear",
        "buffer_init_steps": 1000,
        "tau": 0.005,
        "steps_per_update": 10.0,
        "policy_delay": 2,
        "target_noise": 0.2,
        "noise_clip": 0.5,
        "exploration_noise": 0.1,
        "save_replay_buffer": True,
        "reward_signal_steps_per_update": 1.0,
        "network_settings": {
            "normalize": True,
            "hidden_units": 512,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "goal_conditioning_type": "hyper",
            "deterministic": False,
            "memory": {
                "sequence_length": 64,
                "memory_size": 256
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.995,
                "strength": 1.0,
                "network_settings": {
                    "normalize": False,
                    "hidden_units": 128,
                    "num_layers": 2,
                    "vis_encode_type": "simple",
                    "memory": None,
                    "goal_conditioning_type": "hyper",
                    "deterministic": False
                }
            }
        }
    },
    "ppo_et": {
        "batch_size": 2048,
        "buffer_size": 20480,
        "learning_rate": 0.00035,
        "beta": 0.005,
        "epsilon": 0.3,
        "lambd": 0.95,
        "num_epoch": 4,
        "shared_critic": False,
        "learning_rate_schedule": "linear",
        "beta_schedule": "linear",
        "epsilon_schedule": "linear",
        "entropy_temperature": 1.0,
        "entropy_temperature_schedule": "constant",
        "adaptive_entropy_temperature": True,
        "target_entropy": None,
        "network_settings": {
            "normalize": True,
            "hidden_units": 512,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "goal_conditioning_type": "hyper",
            "deterministic": False,
            "memory": {
                "sequence_length": 64,
                "memory_size": 256
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.995,
                "strength": 1.0,
                "network_settings": {
                    "normalize": False,
                    "hidden_units": 128,
                    "num_layers": 2,
                    "vis_encode_type": "simple",
                    "memory": None,
                    "goal_conditioning_type": "hyper",
                    "deterministic": False
                }
            }
        }
    },
    "ppo_ce": {
        "batch_size": 2048,
        "buffer_size": 20480,
        "learning_rate": 0.0003,
        "beta": 0.01,
        "epsilon": 0.2,
        "lambd": 0.95,
        "num_epoch": 3,
        "shared_critic": False,
        "learning_rate_schedule": "linear",
        "beta_schedule": "linear",
        "epsilon_schedule": "linear",
        "curiosity_strength": 0.02,
        "curiosity_gamma": 0.995,
        "curiosity_learning_rate": 0.0003,
        "curiosity_hidden_units": 256,
        "curiosity_num_layers": 3,
        "imagination_horizon": 5,
        "use_imagination_augmented": False,
        "curiosity_loss_weight": 1.0,
        "network_settings": {
            "normalize": True,
            "hidden_units": 512,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "goal_conditioning_type": "hyper",
            "deterministic": False,
            "memory": {
                "sequence_length": 64,
                "memory_size": 256
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.995,
                "strength": 1.0,
                "network_settings": {
                    "normalize": False,
                    "hidden_units": 128,
                    "num_layers": 2,
                    "vis_encode_type": "simple",
                    "memory": None,
                    "goal_conditioning_type": "hyper",
                    "deterministic": False
                }
            }
        }
    },
    "tdsac": {
        "batch_size": 2048,
        "buffer_size": 200000,
        "learning_rate": 0.0003,
        "learning_rate_schedule": "linear",
        "buffer_init_steps": 1000,
        "tau": 0.005,
        "steps_per_update": 1.0,
        "save_replay_buffer": True,
        "init_entcoef": 0.1,
        "reward_signal_steps_per_update": 1.0,
        "network_settings": {
            "normalize": True,
            "hidden_units": 512,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "goal_conditioning_type": "hyper",
            "deterministic": False,
            "memory": {
                "sequence_length": 64,
                "memory_size": 256
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.995,
                "strength": 1.0,
                "network_settings": {
                    "normalize": False,
                    "hidden_units": 128,
                    "num_layers": 2,
                    "vis_encode_type": "simple",
                    "memory": None,
                    "goal_conditioning_type": "hyper",
                    "deterministic": False
                }
            }
        }
    },
    "tqc": {
        "batch_size": 2048,
        "buffer_size": 200000,
        "learning_rate": 0.00035,
        "learning_rate_schedule": "linear",
        "buffer_init_steps": 1000,
        "tau": 0.005,
        "steps_per_update": 1.0,
        "save_replay_buffer": True,
        "init_entcoef": 1.0,
        "reward_signal_steps_per_update": 4.0,
        "n_quantiles": 25,
        "n_to_drop": 2,
        "network_settings": {
            "normalize": True,
            "hidden_units": 512,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "goal_conditioning_type": "hyper",
            "deterministic": False,
            "memory": {
                "sequence_length": 64,
                "memory_size": 256
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.995,
                "strength": 1.0,
                "network_settings": {
                    "normalize": False,
                    "hidden_units": 128,
                    "num_layers": 2,
                    "vis_encode_type": "simple",
                    "memory": None,
                    "goal_conditioning_type": "hyper",
                    "deterministic": False
                }
            }
        }
    },
    "drqv2": {
        "batch_size": 1024,
        "buffer_size": 100000,
        "learning_rate": 0.00035,
        "learning_rate_schedule": "linear",
        "buffer_init_steps": 1000,
        "tau": 0.005,
        "steps_per_update": 1.0,
        "save_replay_buffer": True,
        "init_entcoef": 1.0,
        "reward_signal_steps_per_update": 1.0,
        "image_pad": 4,
        "network_settings": {
            "normalize": True,
            "hidden_units": 512,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "goal_conditioning_type": "hyper",
            "deterministic": False,
            "memory": {
                "sequence_length": 64,
                "memory_size": 256
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.995,
                "strength": 1.0,
                "network_settings": {
                    "normalize": False,
                    "hidden_units": 128,
                    "num_layers": 2,
                    "vis_encode_type": "simple",
                    "memory": None,
                    "goal_conditioning_type": "hyper",
                    "deterministic": False
                }
            }
        }
    },
    "dcac": {
        "batch_size": 2048,
        "buffer_size": 200000,
        "learning_rate": 0.00035,
        "learning_rate_schedule": "linear",
        "buffer_init_steps": 1000,
        "tau": 0.005,
        "steps_per_update": 1.0,
        "save_replay_buffer": True,
        "init_entcoef": 1.0,
        "reward_signal_steps_per_update": 4.0,
        "destructive_threshold": 0.0,
        "network_settings": {
            "normalize": True,
            "hidden_units": 512,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "goal_conditioning_type": "hyper",
            "deterministic": False,
            "memory": {
                "sequence_length": 64,
                "memory_size": 256
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.995,
                "strength": 1.0,
                "network_settings": {
                    "normalize": False,
                    "hidden_units": 128,
                    "num_layers": 2,
                    "vis_encode_type": "simple",
                    "memory": None,
                    "goal_conditioning_type": "hyper",
                    "deterministic": False
                }
            }
        }
    },
    "crossq": {
        "batch_size": 2048,
        "buffer_size": 200000,
        "learning_rate": 0.00035,
        "learning_rate_schedule": "linear",
        "buffer_init_steps": 1000,
        "tau": 0.005,
        "steps_per_update": 1.0,
        "save_replay_buffer": True,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "noise_clip": 0.5,
        "reward_signal_steps_per_update": 4.0,
        "network_settings": {
            "normalize": True,
            "hidden_units": 512,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "goal_conditioning_type": "hyper",
            "deterministic": False,
            "memory": {
                "sequence_length": 64,
                "memory_size": 256
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.995,
                "strength": 1.0,
                "network_settings": {
                    "normalize": False,
                    "hidden_units": 128,
                    "num_layers": 2,
                    "vis_encode_type": "simple",
                    "memory": None,
                    "goal_conditioning_type": "hyper",
                    "deterministic": False
                }
            }
        }
    },
    "dreamer": {
        "batch_size": 1024,
        "buffer_size": 100000,
        "learning_rate": 1e-4,
        "learning_rate_schedule": "constant",
        "buffer_init_steps": 1000,
        "steps_per_update": 1.0,
        "save_replay_buffer": True,
        "reward_signal_steps_per_update": 1.0,
        
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
            "normalize": False,
            "hidden_units": 512,
            "num_layers": 3,
            "vis_encode_type": "simple",
            "goal_conditioning_type": "hyper",
            "deterministic": False,
            "memory": {
                "sequence_length": 64, 
                "memory_size": 512
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.997,
                "strength": 1.0
            }
        }
    }
}
