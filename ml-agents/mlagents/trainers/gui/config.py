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
            "vis_encode_type": "simple"
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
            "vis_encode_type": "simple"
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.95,
                "strength": 0.99
            }
        }
    },
    "td3": {
        "batch_size": 128,
        "buffer_size": 50000,
        "learning_rate": 0.0003,
        "learning_rate_schedule": "constant",
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
            "normalize": False,
            "hidden_units": 128,
            "num_layers": 2,
            "vis_encode_type": "simple",
            "memory": {
                "sequence_length": 64,
                "memory_size": 128
            }
        },
        "reward_signals": {
            "extrinsic": {
                "gamma": 0.99,
                "strength": 1.0
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
    # "tdsac": { ... },
    # "tqc": { ... },
    # "dcac": { ... },
    # "crossq": { ... },
    # "ppo_et": { ... },
    # "ppo_ce": { ... },
    # "sac_ae": { ... },
    # "drqv2": { ... }
}
