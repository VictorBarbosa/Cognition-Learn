from typing import Dict, List, NamedTuple, Optional, Union
import attr
from mlagents.trainers.settings import (
    OffPolicyHyperparamSettings,
    ScheduleType,
)


@attr.s(auto_attribs=True)
class DreamerSettings(OffPolicyHyperparamSettings):
    batch_size: int = 16
    buffer_size: int = 100000
    learning_rate: float = 1e-4
    
    # Dreamer specific
    batch_length: int = 64  # Length of sequence to train World Model
    horizon: int = 15       # Imagination horizon
    
    # Model Architecture
    hidden_units: int = 512
    gru_units: int = 512
    cnn_depth: int = 32
    mlp_layers: int = 3
    
    # Learning Rates
    model_lr: float = 1e-4
    actor_lr: float = 8e-5
    value_lr: float = 8e-5
    
    # Losses
    kl_scale: float = 1.0
    free_nats: float = 3.0
    
    steps_per_update: float = 1.0
    reward_signal_steps_per_update: float = attr.ib()

    @reward_signal_steps_per_update.default
    def _reward_signal_steps_per_update_default(self):
        return self.steps_per_update
