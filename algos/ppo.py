# Code found here: https://github.com/denisyarats/pytorch_sac

import torch

from algos.agent import Agent
from algos.utils import DiagGaussianActor
from logger import logger

from algos import utils


class PPOAgent(Agent):
    """PPO algorithm."""

    def __init__(self, args, obs_preprocessor, teacher, env, device=None, advice_dim=128):
        obs = env.reset()
        self.action_dim = env.action_space.n if args.discrete else env.action_space.shape[0]
        self.action_shape = 1 if args.discrete else self.action_dim
        self.advice_size = 0 if teacher is 'none' else len(obs[teacher])
        self.advice_dim = 0 if self.advice_size == 0 else advice_dim
        if device is None:
           device = 'cuda' if torch.cuda.is_available() else 'cpu'
        super().__init__(args, obs_preprocessor, teacher, env, device=device, advice_size=self.advice_size,
                         advice_dim=128)
        obs_dim = (128 + self.advice_dim) if args.image_obs else (len(obs['obs'].flatten()) + self.advice_dim)
        self.critic = utils.mlp(obs_dim, args.hidden_dim, 1, 2).to(self.device)
        self.actor = DiagGaussianActor(obs_dim, self.action_dim, discrete=args.discrete, hidden_dim=args.hidden_dim).to(
            self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=self.args.optim_eps)
        self.train()

    def log_critic(self, tag, critic_loss, value, collected_value, collected_return, obs, grad_norm, clip):
        logger.logkv(f'{tag}/Value_loss', utils.to_np(critic_loss))
        logger.logkv(f'{tag}/V_mean', utils.to_np(value.mean()))
        logger.logkv(f'{tag}/Return', utils.to_np(collected_return.mean()))
        logger.logkv(f'{tag}/Collected_value', utils.to_np(collected_value.mean()))
        logger.logkv(f'{tag}/V_std', utils.to_np(value.std()))
        logger.logkv(f'{tag}/obs_min', utils.to_np(obs.min()))
        logger.logkv(f'{tag}/obs_max', utils.to_np(obs.max()))
        logger.logkv('Train/Grad_norm_critic', utils.to_np(grad_norm))
        logger.logkv('Train/ValueClip', utils.to_np(clip.mean()))

    def log_actor(self, actor_loss, dist, value, policy_loss, control_penalty, action, log_prob, recon_loss,
                  grad_norm, clip, ratio):
        logger.logkv('Train/LogProb', utils.to_np(log_prob).mean())
        logger.logkv('Train/Loss', utils.to_np(actor_loss))
        logger.logkv('Train/Entropy', utils.to_np(dist.entropy().mean()))
        logger.logkv('Train/Entropy_Loss',  - self.args.entropy_coef * utils.to_np(dist.entropy().mean()))
        logger.logkv('Train/Entropy_loss',  - self.args.entropy_coef * utils.to_np(dist.entropy().mean()))
        logger.logkv('Train/V', utils.to_np(value.mean()))
        logger.logkv('Train/policy_loss', utils.to_np(policy_loss))
        logger.logkv('Train/Policy_loss', utils.to_np(policy_loss))
        logger.logkv('Train/control_penalty', utils.to_np(control_penalty))
        logger.logkv('Train/recon_loss', utils.to_np(recon_loss))
        logger.logkv('Train/Grad_norm_actor', utils.to_np(grad_norm))
        logger.logkv('Train/PolicyClip', utils.to_np(clip.mean()))
        logger.logkv('Train/Ratio', utils.to_np(ratio.mean()))
        if not self.args.discrete:
            logger.logkv('Train/Action_abs_mean', utils.to_np(torch.abs(dist.loc).mean()))
            logger.logkv('Train/Action_std', utils.to_np(dist.scale.mean()))
        logger.logkv('Train/Action_magnitude', utils.to_np(action.float().norm(2, dim=-1).mean()))
        logger.logkv('Train/Action_magnitude_L1', utils.to_np(action.float().norm(1, dim=-1).mean()))
        logger.logkv('Train/Action_max', utils.to_np(action.float().max(dim=-1)[0].mean()))

    def update_actor(self, obs, batch, advice=None, no_advice_obs=None, next_obs=None):
        assert len(obs.shape) == 2
        batch_size = len(obs)
        import time
        t = time.time()
        dist = self.actor(obs)
        logger.logkv('Time/B_actor_time', time.time() - t)
        t = time.time()
        entropy = dist.entropy().mean()
        action = batch.action
        assert action.shape == (batch_size, self.action_shape), (action.shape, batch_size, self.action_shape)
        if self.args.discrete:
            new_log_prob = dist.log_prob(action[:, 0])
        else:
            new_log_prob = dist.log_prob(action).sum(-1)
        assert batch.log_prob.shape == new_log_prob.shape == batch.advantage.shape == (batch_size, )
        
        # Ratio is pi(a|s) / pi_old(a|s), where pi is the current policy and pi_old is the policy which generated the data
        # Since we compute it in log space, the ratio is a subtration
        ratio = torch.exp(new_log_prob - batch.log_prob)
        
        # PPO loss is the min of the clipped and unclipped ratio * the advantage
        surrr1 = ratio * batch.advantage
        surrr2 = torch.clamp(ratio, 1.0 - self.args.clip_eps, 1.0 + self.args.clip_eps) * batch.advantage
        
        # control penalty (only used w continuous actions) discourages large actions
        if self.args.discrete:
            control_penalty = torch.zeros(1, device=ratio.device).mean()
        else:
            control_penalty = dist.rsample().float().norm(2, dim=-1).mean()
        policy_loss = -torch.min(surrr1, surrr2).mean()
        
        # Reconstruction loss has the agent reconstruct the advice (not used by default)
        if self.args.recon_coef > 0:
            recon_loss = self.compute_recon_loss(dist, no_advice_obs, advice)
        else:
            recon_loss = torch.zeros(1, device=ratio.device).mean()
            
        # Actor loss is the sum of the policy loss, the entropy loss, and the reconstruction loss
        # Entropy loss is negative because we want to encourage entropy to encourage exploration
        actor_loss = policy_loss - self.args.entropy_coef * entropy + self.args.control_penalty * control_penalty + \
                     self.args.recon_coef * recon_loss
        logger.logkv('Time/B_compute_loss_time', time.time() - t)
        t = time.time()
        t = time.time()
        clip = surrr1 - surrr2
        self.log_actor(actor_loss, dist, batch.value, policy_loss, control_penalty, action, new_log_prob, recon_loss,
                       torch.zeros_like(actor_loss), clip, ratio)
        logger.logkv('Time/B_log_time', time.time() - t)

        assert len(obs.shape) == 2
        assert obs.shape == next_obs.shape
        collected_value = batch.value
        collected_return = batch.returnn
        
        # Critic predicts the value of the state (expected return from that state)
        value = self.critic(obs)[:, 0]  # (batch, )
        assert value.shape == collected_value.shape == collected_return.shape, \
            (value.shape, collected_value.shape, collected_return.shape)
        assert value.shape == torch.Size((len(obs), )), (value.shape, len(obs))
        
        # Clip the value function to prevent it from going too far from the value collected with the batch
        value_clipped = collected_value + torch.clamp(value - collected_value, -self.args.clip_eps, self.args.clip_eps)
        
        # Critic loss is the max of the clipped and unclipped squared error between the value and target value (i.e. return)
        surr1 = (value - collected_return).pow(2)
        surr2 = (value_clipped - collected_return).pow(2)
        critic_loss = torch.max(surr1, surr2).mean()

        tag = 'Train'
        # Zero gradients from last batch
        self.optimizer.zero_grad()
        # Combine actor and critic losses. .5 is a random scaling factor, but seems common (e.g. https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py)
        loss = actor_loss + .5 * critic_loss
        # Backpropagate
        loss.backward()
        # Clip gradients to prevent them from exploding
        torch.nn.utils.clip_grad_norm_(self.parameters(), .5)
        # Update
        self.optimizer.step()
        clip = surr1 - surr2
        grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.parameters() if p.grad is not None) ** 0.5
        self.log_critic(tag, critic_loss, value, collected_value, collected_return, obs, grad_norm, clip)

    def act(self, obs, sample=False, instr_dropout_prob=0):
        obs, addl_obs = self.format_obs(obs, instr_dropout_prob=instr_dropout_prob)
        dist = self.actor(obs)
        argmax_action = dist.probs.argmax(dim=1) if self.args.discrete else dist.mean
        action = dist.sample() if sample else argmax_action
        value = self.critic(obs)
        agent_info = {'argmax_action': argmax_action, 'dist': dist, 'value': value, 'addl_obs': addl_obs}
        if len(action.shape) == 1:  # Make sure discrete envs still have an action_dim dimension
            action = action.unsqueeze(1)
        return action, agent_info

