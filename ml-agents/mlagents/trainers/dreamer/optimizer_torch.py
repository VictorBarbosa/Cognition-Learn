import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import attr

from mlagents.torch_utils import torch, nn, default_device
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.optimizer.torch_optimizer import TorchOptimizer
from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents_envs.timers import timed
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.dreamer.settings import DreamerSettings
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.torch_entities.networks import NetworkBody
from mlagents.trainers.torch_entities.layers import linear_layer, Initialization

logger = get_logger(__name__)

# --- Helper Functions (Symlog) ---
def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

# --- Components ---

class RSSM(nn.Module):
    def __init__(self, embed_dim, action_dim, hidden_dim=512, stoch_dim=32, discrete_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stoch_dim = stoch_dim
        self.discrete_dim = discrete_dim
        
        # Deterministic Path (GRU)
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Prior (Predicts state from past)
        self.prior_mlp = nn.Sequential(
            linear_layer(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            linear_layer(hidden_dim, stoch_dim * discrete_dim)
        )
        
        # Posterior (Observes reality)
        self.posterior_mlp = nn.Sequential(
            linear_layer(hidden_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            linear_layer(hidden_dim, stoch_dim * discrete_dim)
        )
        
        # Input processing
        self.action_process = linear_layer(action_dim, hidden_dim)
        self.state_process = linear_layer(stoch_dim * discrete_dim, hidden_dim)

    def initial_state(self, batch_size):
        device = default_device()
        return dict(
            deter=torch.zeros(batch_size, self.hidden_dim, device=device),
            stoch=torch.zeros(batch_size, self.stoch_dim * self.discrete_dim, device=device),
            logits=torch.zeros(batch_size, self.stoch_dim, self.discrete_dim, device=device)
        )

    def observe(self, embed, action, state=None):
        if state is None:
            state = self.initial_state(embed.shape[0])
        
        # 1. Deterministic Step (Time t)
        # Combine prev_stoch and prev_action
        x = self.state_process(state['stoch']) + self.action_process(action)
        deter = self.gru(x, state['deter'])
        
        # 2. Posterior Step (Correction with Obs)
        # Concatenate deter and current observation
        x_post = torch.cat([deter, embed], dim=-1)
        logits_post = self.posterior_mlp(x_post).reshape(-1, self.stoch_dim, self.discrete_dim)
        stoch_post = self.get_stoch(logits_post)
        
        # 3. Prior (for Loss calculation)
        logits_prior = self.prior_mlp(deter).reshape(-1, self.stoch_dim, self.discrete_dim)
        
        post_state = {'deter': deter, 'stoch': stoch_post, 'logits': logits_post}
        prior_state = {'deter': deter, 'stoch': stoch_post, 'logits': logits_prior} # Note: stoch is from posterior
        
        return post_state, prior_state

    def imagine(self, action, state):
        # 1. Deterministic Step
        x = self.state_process(state['stoch']) + self.action_process(action)
        deter = self.gru(x, state['deter'])
        
        # 2. Prior Step (Prediction without Obs)
        logits = self.prior_mlp(deter).reshape(-1, self.stoch_dim, self.discrete_dim)
        stoch = self.get_stoch(logits)
        
        return {'deter': deter, 'stoch': stoch, 'logits': logits}

    def get_stoch(self, logits):
        # Straight-Through Gumbel Softmax (Discrete Autoencoder)
        dist = torch.distributions.OneHotCategorical(logits=logits)
        stoch = dist.sample()
        # Add straight-through gradients
        stoch = stoch + dist.probs - dist.probs.detach()
        return stoch.flatten(1)

class WorldModel(nn.Module):
    def __init__(self, obs_specs, action_spec, settings: DreamerSettings):
        super().__init__()
        self.settings = settings
        
        # Encoders (using ML-Agents standard encoders)
        self.encoder = NetworkBody(obs_specs, settings.network_settings)
        embed_dim = self.encoder.h_size
        
        # Action Dim
        self.action_dim = action_spec.continuous_size + sum(action_spec.discrete_branches)
        
        # RSSM
        self.rssm = RSSM(embed_dim, self.action_dim, settings.hidden_units)
        feat_dim = settings.hidden_units + (32 * 32) # hidden + stoch*discrete
        
        # Heads
        self.reward_head = nn.Sequential(
            linear_layer(feat_dim, 256), nn.SiLU(),
            linear_layer(256, 1) # Symlog Reward
        )
        self.cont_head = nn.Sequential(
            linear_layer(feat_dim, 256), nn.SiLU(),
            linear_layer(256, 1), nn.Sigmoid() # Continue prob (Bernoulli)
        )
        # Decoder (Optional for reconstruction loss - simplified here to just feature match if needed, 
        # but standard Dreamer decodes to Pixels. 
        # For simplicity in ML-Agents integration, we will skip pixel decoding in this MVP 
        # and rely on Reward/Continue + KL for learning representations)
        
        self.opt = torch.optim.Adam(self.parameters(), lr=settings.model_lr)

    def forward(self, obs_list, action, state=None):
        embed, _ = self.encoder(obs_list, sequence_length=1)
        post, prior = self.rssm.observe(embed, action, state)
        return post, prior

class ActorCritic(nn.Module):
    def __init__(self, feat_dim, action_spec, settings: DreamerSettings):
        super().__init__()
        self.action_spec = action_spec
        
        # Actor
        self.actor = nn.Sequential(
            linear_layer(feat_dim, settings.hidden_units), nn.LayerNorm(settings.hidden_units), nn.SiLU(),
            linear_layer(settings.hidden_units, settings.hidden_units), nn.LayerNorm(settings.hidden_units), nn.SiLU(),
            linear_layer(settings.hidden_units, action_spec.continuous_size * 2) # Mu, Std
        )
        
        # Critic (Value) - Predicts Symlog(Return)
        self.critic = nn.Sequential(
            linear_layer(feat_dim, settings.hidden_units), nn.LayerNorm(settings.hidden_units), nn.SiLU(),
            linear_layer(settings.hidden_units, settings.hidden_units), nn.LayerNorm(settings.hidden_units), nn.SiLU(),
            linear_layer(settings.hidden_units, 1)
        )
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=settings.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=settings.value_lr)

    def get_action(self, feat):
        out = self.actor(feat)
        mu, std = torch.chunk(out, 2, dim=-1)
        std = torch.sigmoid(std) + 0.1
        dist = torch.distributions.Normal(torch.tanh(mu), std)
        action = dist.rsample()
        return action, dist

# --- Optimizer ---

class TorchDreamerOptimizer(TorchOptimizer):
    def __init__(self, policy: TorchPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        self.settings = cast(DreamerSettings, trainer_settings.hyperparameters)
        
        # Instantiate Modules
        self.world_model = WorldModel(
            policy.behavior_spec.observation_specs,
            policy.behavior_spec.action_spec,
            self.settings
        ).to(default_device())
        
        feat_dim = self.settings.hidden_units + (32 * 32)
        self.ac = ActorCritic(
            feat_dim,
            policy.behavior_spec.action_spec,
            self.settings
        ).to(default_device())
        
        self._step = 0

    @property
    def critic(self):
        return self.ac.critic # Dummy implementation for base class requirements

    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:
        self._step += 1
        
        # 1. Prepare Data
        # ML-Agents buffers are flattened [Batch * Seq, Features] if not LSTM?
        # But we need sequential data [Batch, Seq, Features]
        # We assume batch comes from `sample_mini_batch` with sequence_length=batch_length
        
        n_obs = len(self.policy.behavior_spec.observation_specs)
        current_obs = ObsUtil.from_buffer(batch, n_obs)
        # Convert to tensors [Batch, Seq, Feat]
        # Ops... ObsUtil flattens. We need to reshape.
        
        batch_size = self.settings.batch_size
        seq_len = self.settings.batch_length
        
        # Verify sizes
        if batch.num_experiences < batch_size * seq_len:
            return {} # Not enough data
            
        # Helper to reshape flattened buffer to [Batch, Seq, ...]
        def reshape_tensor(tensor):
            t = ModelUtils.list_to_tensor(tensor)
            # Assuming batch is structured as [Seq1, Seq2, ...]
            # We want [Batch, Seq, ...]
            return t.reshape(batch_size, seq_len, -1)

        # Obs
        obs_list = [reshape_tensor(o) for o in current_obs]
        
        # Actions
        actions = ModelUtils.list_to_tensor(batch[BufferKey.CONTINUOUS_ACTION])
        actions = actions.reshape(batch_size, seq_len, -1)
        
        # Rewards
        rewards = ModelUtils.list_to_tensor(batch[BufferKey.ENVIRONMENT_REWARDS])
        rewards = rewards.reshape(batch_size, seq_len, 1)
        
        # Dones
        dones = ModelUtils.list_to_tensor(batch[BufferKey.DONE])
        dones = dones.reshape(batch_size, seq_len, 1)
        
        # --- Train World Model ---
        self.world_model.opt.zero_grad()
        
        state = self.world_model.rssm.initial_state(batch_size)
        kl_losses = []
        rew_losses = []
        cont_losses = []
        
        # Unroll Sequence
        # Flatten time and batch for encoder? No, iterate time
        for t in range(seq_len):
            # Encode Obs
            t_obs = [o[:, t] for o in obs_list]
            t_act = actions[:, t]
            t_rew = rewards[:, t]
            t_cont = 1.0 - dones[:, t]
            
            # Encoder forward
            embed, _ = self.world_model.encoder(t_obs) # [Batch, Embed]
            
            # RSSM Observe
            post, prior = self.world_model.rssm.observe(embed, t_act, state)
            state = post
            
            # Features
            feat = torch.cat([post['deter'], post['stoch']], dim=-1)
            
            # Heads
            pred_rew = self.world_model.reward_head(feat)
            pred_cont = self.world_model.cont_head(feat)
            
            # Losses
            # 1. Reward (Symlog MSE)
            rew_losses.append(torch.mean((pred_rew - symlog(t_rew))**2))
            
            # 2. Continue (BCE)
            cont_losses.append(torch.nn.functional.binary_cross_entropy(pred_cont, t_cont))
            
            # 3. KL Divergence (Dynamic Learning)
            # KL between Post (Observed) and Prior (Predicted)
            # Categorical KL
            p = post['logits'].softmax(-1)
            q = prior['logits'].softmax(-1)
            # Dynamic loss (stop grad on prior/q to train representation)
            # Representation loss (stop grad on post/p to train dynamics)
            # Dreamer V3 balances this. Simplified here:
            kl = torch.sum(p * (torch.log(p + 1e-8) - torch.log(q + 1e-8)), dim=-1).mean()
            kl_losses.append(torch.max(kl, torch.tensor(self.settings.free_nats).to(default_device())))

        wm_loss = sum(rew_losses) + sum(cont_losses) + self.settings.kl_scale * sum(kl_losses)
        wm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.world_model.opt.step()
        
        # --- Train Actor Critic (Imagination) ---
        # Start from posterior states of the sequence
        # Flatten batch and seq to treat all visited states as starting points
        with torch.no_grad():
            start_deter = state['deter'].detach() # Use last state of sequence
            start_stoch = state['stoch'].detach()
            start_state = {'deter': start_deter, 'stoch': start_stoch}
        
        self.ac.actor_opt.zero_grad()
        self.ac.critic_opt.zero_grad()
        
        imag_feats = []
        imag_rews = []
        imag_conts = []
        imag_vals = []
        imag_log_probs = []
        
        curr_state = start_state
        
        # Dream Horizon
        for t in range(self.settings.horizon):
            feat = torch.cat([curr_state['deter'], curr_state['stoch']], dim=-1)
            imag_feats.append(feat)
            
            # Action
            action, dist = self.ac.get_action(feat.detach())
            imag_log_probs.append(dist.log_prob(action).sum(-1))
            
            # Step World Model (Prior only)
            curr_state = self.world_model.rssm.imagine(action, curr_state)
            
            # Predict Rewards/Values/Cont
            next_feat = torch.cat([curr_state['deter'], curr_state['stoch']], dim=-1)
            rew = self.world_model.reward_head(next_feat)
            cont = self.world_model.cont_head(next_feat)
            val = self.ac.critic(next_feat)
            
            imag_rews.append(rew)
            imag_conts.append(cont)
            imag_vals.append(val)

        # Compute Lambda Returns (Dreamer V3 value estimation)
        # Simplified GAE/Lambda return
        returns = []
        last_val = imag_vals[-1]
        for t in reversed(range(self.settings.horizon)):
            r = imag_rews[t]
            c = imag_conts[t]
            v = imag_vals[t]
            # y = r + gamma * c * next_v
            # Dreamer uses symexp transform for returns
            # Keep it simple: standard lambda return in symlog space?
            # V3: v_target = symlog(symexp(r) + gamma * c * symexp(next_v))
            
            next_v = last_val if t == self.settings.horizon - 1 else returns[0] # Returns is prepended, so 0 is next
            
            # Inverse Symlog (Symexp) for accumulation
            # target = symexp(r) + 0.99 * c * symexp(next_v)
            # return_val = symlog(target)
            
            # Using simple additive for stability in this MVP
            target = r + 0.99 * c * next_v
            returns.insert(0, target)
            
        # Actor Loss (Reinforce/Dynamics Backprop)
        # Dreamer V3 uses dynamics backprop for continuous.
        # But we detached world model. So we rely on Reinforce + Value baseline?
        # Actually V3 backprops through the model.
        # Re-implementing full dynamics backprop requires retaining graph in 'imagine'.
        # For this MVP, we use Policy Gradient on the imagined returns.
        
        actor_loss = 0
        critic_loss = 0
        
        for t in range(self.settings.horizon):
            # Critic Loss: MSE between predicted value and lambda return
            critic_loss += torch.mean((imag_vals[t] - returns[t].detach())**2)
            
            # Actor Loss: Maximize Value (Dynamics Backprop would be -returns[t].mean())
            # Reinforce: -log_prob * (return - baseline)
            adv = returns[t].detach() - imag_vals[t].detach()
            actor_loss += -torch.mean(imag_log_probs[t] * adv)
            
        (actor_loss + critic_loss).backward()
        self.ac.actor_opt.step()
        self.ac.critic_opt.step()
        
        return {
            "Dreamer/WM Loss": wm_loss.item(),
            "Dreamer/Actor Loss": actor_loss.item(),
            "Dreamer/Critic Loss": critic_loss.item(),
            "Dreamer/Pred Reward": torch.stack(imag_rews).mean().item()
        }

    def get_modules(self):
        return {
            "WorldModel": self.world_model,
            "ActorCritic": self.ac
        }
