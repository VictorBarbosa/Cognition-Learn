# Default Hyperparameters for each algorithm

ALGO_DEFAULTS = {
    "ppo": {
        "batch_size": 1024,
        "buffer_size": 10240,
        "learning_rate": 0.0003,
        "beta": 0.005,
        "epsilon": 0.2,
        "lambd": 0.95,
        "num_epoch": 3,
        "shared_critic": False,
        "learning_rate_schedule": "linear",
        "beta_schedule": "linear",
        "epsilon_schedule": "linear",
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
    "sac": {
        "batch_size": 128,
        "buffer_size": 50000,
        "learning_rate": 0.0003,
        "learning_rate_schedule": "constant",
        "buffer_init_steps": 0,
        "tau": 0.005,
        "steps_per_update": 1.0,
        "save_replay_buffer": False,
        "init_entcoef": 1.0,
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
    # "td3": { ... },
    # "tdsac": { ... },
    # "tqc": { ... },
    # "dcac": { ... },
    # "crossq": { ... },
    # "ppo_et": { ... },
    # "ppo_ce": { ... },
    # "sac_ae": { ... },
    # "drqv2": { ... }
}
