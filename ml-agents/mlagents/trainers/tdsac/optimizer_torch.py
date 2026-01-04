import numpy as np
from typing import Dict, List, NamedTuple, cast, Tuple, Optional
import attr

from mlagents.torch_utils import torch, nn, default_device

from mlagents_envs.logging_util import get_logger
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.networks import ValueNetwork, SharedActorCritic
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents_envs.timers import timed
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents.trainers.exception import UnityTrainerException
from mlagents.trainers.settings import TrainerSettings, OffPolicyHyperparamSettings
from contextlib import ExitStack
from mlagents.trainers.trajectory import ObsUtil
from mlagents.trainers.sac.optimizer_torch import TorchSACOptimizer, SACSettings

EPSILON = 1e-6  # Small value to avoid divide by zero

logger = get_logger(__name__)


@attr.s(auto_attribs=True)
class TDSACSettings(SACSettings):
    pass


class TorchTDSACOptimizer(TorchSACOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        # Create Target Q Network (similar to TD3)
        self.target_q_network = TorchSACOptimizer.PolicyValueNetwork(
            self.stream_names,
            self.policy.behavior_spec.observation_specs,
            policy.network_settings,
            self._action_spec,
        )
        ModelUtils.soft_update(self.q_network, self.target_q_network, 1.0)
        self._move_to_device(default_device())

    def _move_to_device(self, device: torch.device) -> None:
        super()._move_to_device(device)
        if hasattr(self, "target_q_network"):
            self.target_q_network.to(device)

    def sac_value_loss(
        self,
        log_probs: ActionLogProbs,
        values: Dict[str, torch.Tensor],
        q1p_out: Dict[str, torch.Tensor],
        q2p_out: Dict[str, torch.Tensor],
        loss_masks: torch.Tensor,
    ) -> torch.Tensor:
        # Override to use Target Q instead of Target V (which SAC uses)
        # TDSAC removes the V-network and uses Target Q directly for bootstrapping
        
        min_policy_qs = {}
        with torch.no_grad():
            _cont_ent_coef = self._log_ent_coef.continuous.exp()
            _disc_ent_coef = self._log_ent_coef.discrete.exp()
            for name in values.keys():
                if self._action_spec.discrete_size <= 0:
                    min_policy_qs[name] = torch.min(q1p_out[name], q2p_out[name])
                else:
                    disc_action_probs = log_probs.all_discrete_tensor.exp()
                    _branched_q1p = ModelUtils.break_into_branches(
                        q1p_out[name] * disc_action_probs,
                        self._action_spec.discrete_branches,
                    )
                    _branched_q2p = ModelUtils.break_into_branches(
                        q2p_out[name] * disc_action_probs,
                        self._action_spec.discrete_branches,
                    )
                    _q1p_mean = torch.mean(
                        torch.stack(
                            [
                                torch.sum(_br, dim=1, keepdim=True)
                                for _br in _branched_q1p
                            ]
                        ),
                        dim=0,
                    )
                    _q2p_mean = torch.mean(
                        torch.stack(
                            [
                                torch.sum(_br, dim=1, keepdim=True)
                                for _br in _branched_q2p
                            ]
                        ),
                        dim=0,
                    )

                    min_policy_qs[name] = torch.min(_q1p_mean, _q2p_mean)

        # TDSAC logic: We don't have a V-network loss because we don't have a V-network.
        # This method is called by SAC update but we should bypass it or return 0 if V-net is removed.
        # However, ML-Agents SAC structure enforces V-loss.
        # But wait, self._critic in SAC IS the V-network.
        # In TDSAC, we should technically remove self._critic training.
        
        # For compatibility with SAC's update loop which calls this, we can return 0 if we don't update V.
        # But wait, if we don't update V, what do we use for bootstrapping?
        # TDSAC uses Target Q for bootstrapping: y = r + gamma * (Min(TargetQ1, TargetQ2) - alpha * log_prob)
        
        # SAC uses: y = r + gamma * TargetV
        # TargetV = V_target(s')
        # V(s) = Min(Q1, Q2) - alpha * log_prob
        
        # TDSAC collapses this:
        # TargetV(s') = Min(TargetQ1(s', a'), TargetQ2(s', a')) - alpha * log_prob(a')
        
        # So we need to override how target_values are calculated in `update`, not just this loss.
        return torch.tensor(0.0, device=default_device(), requires_grad=True)

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        """
        Updates model using buffer.
        :param num_sequences: Number of trajectories in batch.
        :param batch: Experience mini-batch.
        :return: Output from update process.
        """
        rewards = {}
        for name in self.reward_signals:
            rewards[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.rewards_key(name)]
            )
            if rewards[name].dim() == 1:
                rewards[name] = rewards[name].unsqueeze(1)

        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]

        next_obs = ObsUtil.from_buffer_next(batch, n_obs)
        next_obs = [ModelUtils.list_to_tensor(obs) for obs in next_obs]

        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK])
        actions = AgentAction.from_buffer(batch)

        memories_list = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        
        if len(memories_list) > 0:
            memories = torch.stack(memories_list).unsqueeze(0)
        else:
            memories = None

        q_memories = (
            torch.zeros_like(memories) if memories is not None else None
        )

        # Copy normalizers
        self.q_network.q1_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.q_network.q2_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.target_q_network.q1_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        self.target_q_network.q2_network.network_body.copy_normalization(
            self.policy.actor.network_body
        )
        
        sampled_actions, run_out, _, = self.policy.actor.get_action_and_stats(
            current_obs,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )
        log_probs = run_out["log_probs"]
        
        cont_sampled_actions = sampled_actions.continuous_tensor
        cont_actions = actions.continuous_tensor
        
        q1p_out, q2p_out = self.q_network(
            current_obs,
            cont_sampled_actions,
            memories=q_memories,
            sequence_length=self.policy.sequence_length,
            q2_grad=False,
        )
        q1_out, q2_out = self.q_network(
            current_obs,
            cont_actions,
            memories=q_memories,
            sequence_length=self.policy.sequence_length,
        )

        if self._action_spec.discrete_size > 0:
            disc_actions = actions.discrete_tensor
            q1_stream = self._condense_q_streams(q1_out, disc_actions)
            q2_stream = self._condense_q_streams(q2_out, disc_actions)
        else:
            q1_stream, q2_stream = q1_out, q2_out

        # Computing Target Q (TDSAC Logic)
        with torch.no_grad():
            next_action, next_run_out, _ = self.policy.actor.get_action_and_stats(
                next_obs, memories=memories, sequence_length=self.policy.sequence_length, masks=act_masks # Using current masks?
            )
            next_log_probs = next_run_out["log_probs"]
            
            target_q1_out, target_q2_out = self.target_q_network(
                next_obs,
                actions=next_action.continuous_tensor,
                memories=q_memories,
                sequence_length=self.policy.sequence_length,
            )
            
            # Min Target Q
            min_target_q = {}
            _cont_ent_coef = self._log_ent_coef.continuous.exp()
            
            for name in target_q1_out.keys():
                t_q1 = target_q1_out[name]
                t_q2 = target_q2_out[name]

                # Condense if discrete
                if self._action_spec.discrete_size > 0:
                    # Manually gather Q-values for the selected discrete actions
                    d_actions = next_action.discrete_tensor.long()
                    
                    # Fix dimensions: [128, 1, 1] -> [128, 1]
                    if d_actions.dim() > 2:
                        d_actions = d_actions.reshape(d_actions.shape[0], -1)
                    
                    t_q1 = t_q1.gather(1, d_actions)
                    t_q2 = t_q2.gather(1, d_actions)

                min_q = torch.min(t_q1, t_q2)

                # Subtract entropy
                if self._action_spec.continuous_size > 0:
                    ent_bonus = torch.sum(_cont_ent_coef * next_log_probs.continuous_tensor, dim=1, keepdim=True)
                    min_q = min_q - ent_bonus
                min_target_q[name] = min_q

        masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)
        dones = ModelUtils.list_to_tensor(batch[BufferKey.DONE])
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)

        # Use sac_q_loss but pass min_target_q as target_values
        q1_loss, q2_loss = self.sac_q_loss(
            q1_stream, q2_stream, min_target_q, dones, rewards, masks
        )
        
        # We don't train V-network (value_loss is 0)
        value_loss = torch.tensor(0.0).to(default_device())
        
        policy_loss = self.sac_policy_loss(log_probs, q1p_out, masks)
        entropy_loss = self.sac_entropy_loss(log_probs, masks)

        total_value_loss = q1_loss + q2_loss

        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        ModelUtils.update_learning_rate(self.policy_optimizer, decay_lr)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        ModelUtils.update_learning_rate(self.value_optimizer, decay_lr)
        self.value_optimizer.zero_grad()
        total_value_loss.backward()
        self.value_optimizer.step()

        ModelUtils.update_learning_rate(self.entropy_optimizer, decay_lr)
        self.entropy_optimizer.zero_grad()
        entropy_loss.backward()
        self.entropy_optimizer.step()

        # Update target networks
        ModelUtils.soft_update(self.q_network, self.target_q_network, self.tau)
        
        update_stats = {
            "Losses/Policy Loss": policy_loss.item(),
            "Losses/Value Loss": total_value_loss.item(), # Q loss
            "Losses/Q1 Loss": q1_loss.item(),
            "Losses/Q2 Loss": q2_loss.item(),
            "Policy/Discrete Entropy Coeff": torch.mean(
                torch.exp(self._log_ent_coef.discrete)
            ).item(),
            "Policy/Continuous Entropy Coeff": torch.mean(
                torch.exp(self._log_ent_coef.continuous)
            ).item(),
            "Policy/Learning Rate": decay_lr,
        }

        return update_stats

    def get_modules(self):
        modules = super().get_modules()
        modules["Optimizer:target_q_network"] = self.target_q_network
        return modules
