from typing import cast
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.trainer.off_policy_trainer import OffPolicyTrainer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.dreamer.optimizer_torch import TorchDreamerOptimizer, DreamerSettings
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.torch_entities.networks import SimpleActor

logger = get_logger(__name__)

TRAINER_NAME = "dreamer"

class DreamerTrainer(OffPolicyTrainer):
    def __init__(self, behavior_name, reward_buff_cap, trainer_settings, training, load, seed, artifact_path):
        super().__init__(
            behavior_name,
            reward_buff_cap,
            trainer_settings,
            training,
            load,
            seed,
            artifact_path,
        )
        
        self.hyperparameters: DreamerSettings = cast(
            DreamerSettings, trainer_settings.hyperparameters
        )
        # Enforce sequence length for the buffer
        self.policy.sequence_length = self.hyperparameters.batch_length

    def create_optimizer(self) -> TorchDreamerOptimizer:
        return TorchDreamerOptimizer(
            cast(TorchPolicy, self.policy), self.trainer_settings
        )

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchPolicy:
        # We use SimpleActor as a placeholder structure for the Policy object,
        # but the Optimizer replaces the logic with the World Model + ActorCritic.
        # Ideally we'd write a DreamerPolicy, but TorchPolicy is tightly coupled.
        # We just need it to carry the behavior_spec and network settings.
        actor_cls = SimpleActor
        actor_kwargs = {"conditional_sigma": True, "tanh_squash": True}
        
        return TorchPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            actor_kwargs,
        )

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME
