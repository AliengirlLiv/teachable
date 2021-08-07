# Code found here: https://github.com/denisyarats/pytorch_sac

import copy
import math
from torch import distributions as pyd
from torch import distributions

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Normal

from logger import logger

from algos import utils


# From https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class FiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=imm_channels,
            kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(
            in_channels=imm_channels, out_channels=out_features,
            kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.conv1(x))
        x = self.conv2(x)
        weight = self.weight(y).unsqueeze(2).unsqueeze(3)
        bias = self.bias(y).unsqueeze(2).unsqueeze(3)
        out = x * weight + bias
        return F.relu(self.bn2(out))
        # return F.relu(out)


class ImageBOWEmbedding(nn.Module):
    def __init__(self, max_value, embedding_dim):
        super().__init__()
        self.max_value = max_value
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(3 * max_value, embedding_dim)
        self.apply(initialize_parameters)

    def forward(self, inputs):
        offsets = torch.Tensor([0, self.max_value, 2 * self.max_value]).to(inputs.device)
        inputs = (inputs + offsets[None, :, None, None]).long()
        return self.embedding(inputs).sum(1).permute(0, 3, 1, 2)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def log_prob(self, value):
        eps = 1e-3
        value = torch.clamp(value, -1 + eps, 1 - eps)
        return super().log_prob(value)



class DiagGaussianActor(nn.Module):
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, obs_dim, action_dim, hidden_dim=1024, hidden_depth=2, log_std_bounds=(-5, 2), discrete=False):
        super().__init__()

        self.discrete = discrete
        self.log_std_bounds = log_std_bounds
        if not discrete:
            action_dim *= 2
        self.trunk = utils.mlp(obs_dim, hidden_dim, action_dim,
                               hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs):
        if self.discrete:
            logits = self.trunk(obs)
            dist = GumbelSoftmax(temperature=1, logits=logits)
        else:
            mu, log_std = self.trunk(obs).chunk(2, dim=-1)

            # constrain log_std inside [log_std_min, log_std_max]
            # log_std = torch.tanh(log_std)
            log_std_min, log_std_max = self.log_std_bounds
            # log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
            log_std = torch.clamp(log_std, log_std_min, log_std_max)

            std = log_std.exp()

            self.outputs['mu'] = mu
            self.outputs['std'] = std

            dist = Normal(mu, std)
        return dist


class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""

    def __init__(self, obs_dim, action_dim, hidden_dim=1024, hidden_depth=2, repeat_action=1):
        super().__init__()
        self.repeat_action = repeat_action

        self.Q1 = utils.mlp(obs_dim + action_dim * repeat_action, hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(obs_dim + action_dim * repeat_action, hidden_dim, 1, hidden_depth)

        self.outputs = dict()
        self.apply(utils.weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs] + [action] * self.repeat_action, dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2


class InstrEmbedding(nn.Module):
    def __init__(self, args, env):
        # Define instruction embedding
        super().__init__()
        self.word_embedding = nn.Embedding(len(env.vocab()), args.instr_dim)
        self.instr_rnn = nn.GRU(args.instr_dim, args.instr_dim, batch_first=True, bidirectional=False)

        self.controllers = []
        for i in range(args.num_modules):
            mod = FiLM(
                in_features=args.instr_dim,
                out_features=128 if i < args.num_modules - 1 else args.image_dim,
                in_channels=128, imm_channels=128)
            self.controllers.append(mod)
            self.add_module('FiLM_' + str(i), mod)

    def forward(self, obs):
        instruction_vector = obs.instr.long()
        instr_embedding = self._get_instr_embedding(instruction_vector)
        obs.instr = instr_embedding
        return obs

    def _get_instr_embedding(self, instr):
        lengths = (instr != 0).sum(1).long()
        out, _ = self.instr_rnn(self.word_embedding(instr))
        hidden = out[range(len(lengths)), lengths - 1, :]
        return hidden


class ImageEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_conv = nn.Sequential(*[
            ImageBOWEmbedding(147, 128),
            nn.Conv2d(
                in_channels=128, out_channels=32,
                kernel_size=(8, 8), stride=8, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=(3, 3), padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
        ])
        self.film_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # Initialize parameters correctly
        self.apply(initialize_parameters)

    def forward(self, obs):
        inputs = torch.transpose(torch.transpose(obs.obs, 1, 3), 2, 3)
        obs.obs = self.image_conv(inputs).flatten(1)
        return obs


class Agent:
    """SAC algorithm."""

    def __init__(self, args, obs_preprocessor, teacher, env,
                 device='cuda', discount=0.99, batch_size=1024, control_penalty=0, repeat_advice=1,
                 actor_update_frequency=1):
        super().__init__()
        if args.discrete:
            action_dim = env.action_space.n
        else:
            action_dim = env.action_space.shape[0]
            self.action_range = (env.action_space.low, env.action_space.high)
        self.action_dim = action_dim
        self.args = args
        self.obs_preprocessor = obs_preprocessor
        self.teacher = teacher
        self.env = env
        self.device = torch.device(device)
        self.discount = discount
        self.batch_size = batch_size
        self.control_penalty = control_penalty
        self.repeat_advice = repeat_advice
        self.actor_update_frequency = actor_update_frequency

        if args.image_obs:
            self.image_encoder = ImageEmbedding().to(self.device)
        else:
            self.image_encoder = None

        if not args.no_instr:
            self.instr_encoder = InstrEmbedding(args, env).to(self.device)
        else:
            self.instr_encoder = None


    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)


    def act(self, obs, sample=False):
        if not 'advice' in obs:  # unpreprocessed
            obs = self.obs_preprocessor(obs, self.teacher, show_instrs=True)
        if self.image_encoder is not None:
            obs = self.image_encoder(obs)
        if self.instr_encoder is not None:
            obs = self.instr_encoder(obs)
        o = torch.cat([obs.obs.flatten(1)] + [obs.advice] * self.repeat_advice, dim=1).to(self.device)
        dist = self.actor(o)
        if self.args.discrete:
            argmax_action = dist.probs.argmax(dim=1)
            action = dist.sample() if sample else argmax_action
        else:
            action = dist.sample() if sample else dist.mean
            min_val, max_val = self.action_range
            action = torch.maximum(action, torch.FloatTensor(min_val).unsqueeze(0).to(self.device))
            action = torch.minimum(action, torch.FloatTensor(max_val).unsqueeze(0).to(self.device))
            argmax_action = dist.mean
        agent_info = {
            'argmax_action': argmax_action,
            'dist': dist,
        }
        return action, agent_info

    def log_rl(self, val_batch):
        self.train(False)
        with torch.no_grad():
            obs = val_batch.obs
            next_obs = val_batch.next_obs
            obs = self.obs_preprocessor(obs, self.teacher, show_instrs=True)
            next_obs = self.obs_preprocessor(next_obs, self.teacher, show_instrs=True)
            if self.image_encoder is not None:
                obs = self.image_encoder(obs)
                next_obs = self.image_encoder(next_obs)
            if self.instr_encoder is not None:
                obs = self.instr_encoder(obs)
                next_obs = self.instr_encoder(next_obs)
            obs = torch.cat([obs.obs.flatten(1)] + [obs.advice] * self.repeat_advice, dim=1).to(self.device)
            next_obs = torch.cat([next_obs.obs.flatten(1)] + [next_obs.advice] * self.repeat_advice, dim=1).to(
                self.device)
            self.update_critic(obs, next_obs, val_batch, train=False)

    def optimize_policy(self, batch, step):
        obs = batch.obs
        reward = batch.reward.unsqueeze(1)
        try:
            next_obs = batch.next_obs
        except AttributeError:
            next_obs = batch.env_infos.next_obs
        obs = self.obs_preprocessor(obs, self.teacher, show_instrs=True)
        preprocessed_obs = copy.deepcopy(obs)
        next_obs = self.obs_preprocessor(next_obs, self.teacher, show_instrs=True)
        if self.image_encoder is not None:
            obs = self.image_encoder(obs)
            next_obs = self.image_encoder(next_obs)
        if self.instr_encoder is not None:
            obs = self.instr_encoder(obs)
            next_obs = self.instr_encoder(next_obs)
        obs = torch.cat([obs.obs.flatten(1)] + [obs.advice] * self.repeat_advice, dim=1).to(self.device)
        next_obs = torch.cat([next_obs.obs.flatten(1)] + [next_obs.advice] * self.repeat_advice, dim=1).to(self.device)

        logger.logkv('train/batch_reward', utils.to_np(reward.mean()))

        self.update_critic(obs, next_obs, batch, step)

        if step % self.actor_update_frequency == 0:
            if self.image_encoder is not None:
                preprocessed_obs = self.image_encoder(preprocessed_obs)
            if self.instr_encoder is not None:
                preprocessed_obs = self.instr_encoder(preprocessed_obs)
            obs = torch.cat([preprocessed_obs.obs.flatten(1)] + [preprocessed_obs.advice] * self.repeat_advice, dim=1).to(self.device)
            self.update_actor(obs, batch)

    def update_critic(self, obs, next_obs, batch, step):
        raise NotImplementedError

    def update_actor(self, obs, batch):
        raise NotImplementedError

    def save(self, model_dir):
        torch.save(self.actor.state_dict(), f'{model_dir}_actor.pt')
        torch.save(self.critic.state_dict(), f'{model_dir}_critic.pt')
        if self.image_encoder is not None:
            torch.save(self.image_encoder.state_dict(), f'{model_dir}_image_encoder.pt')
        if self.instr_encoder is not None:
            torch.save(self.instr_encoder.state_dict(), f'{model_dir}_instr_encoder.pt')

    def load(self, model_dir):
        self.actor.load_state_dict(torch.load(f'{model_dir}_actor.pt'))
        self.critic.load_state_dict(torch.load(f'{model_dir}_critic.pt'))
        if self.image_encoder is not None:
            self.image_encoder.load_state_dict(torch.load(f'{model_dir}_image_encoder.pt'))
        if self.instr_encoder is not None:
            self.instr_encoder.load_state_dict(torch.load(f'{model_dir}_instr_encoder.pt'))

    def reset(self, *args, **kwargs):
        pass

    def get_actions(self, obs, training=False):
        self.train(training=training)
        action, agent_dict = self.act(obs, sample=True)
        return utils.to_np(action[0]), agent_dict

    def log_distill(self, action_true, action_teacher, entropy, policy_loss, loss, dist, train):
        if self.args.discrete:
            action_pred = dist.probs.max(1, keepdim=False)[1]  # argmax action
            avg_mean_dist = -1
            avg_std = -1
        else:  # Continuous env
            action_pred = dist.mean
            avg_std = dist.scale.mean()
            avg_mean_dist = torch.abs(action_pred - action_true).mean()
        log = {
            'Loss': float(policy_loss),
            'Accuracy': float((action_pred == action_true).sum()) / len(action_pred),
        }
        train_str = 'Train' if train else 'Val'
        logger.logkv(f"Distill/Entropy_Loss_{train_str}", float(entropy * self.args.entropy_coef))
        logger.logkv(f"Distill/Entropy_{train_str}", float(entropy))
        logger.logkv(f"Distill/Loss_{train_str}", float(policy_loss))
        logger.logkv(f"Distill/TotalLoss_{train_str}", float(loss))
        logger.logkv(f"Distill/Accuracy_{train_str}", float((action_pred == action_true).sum()) / len(action_pred))
        logger.logkv(f"Distill/Mean_Dist_{train_str}", float(avg_mean_dist))
        logger.logkv(f"Distill/Std_{train_str}", float(avg_std))
        return log

    def preprocess_distill(self, batch, source):
        obss = batch.obs
        if source == 'teacher':
            action_true = batch.teacher_action
            if len(action_true.shape) == 2 and action_true.shape[1] == 1:
                action_true = action_true[:, 0]
        elif source == 'agent':
            action_true = batch.action
        elif source == 'agent_probs':
            if self.args.discrete == False:
                raise NotImplementedError('Action probs not implemented for continuous envs.')
            action_true = batch.argmax_action
        if not source == 'agent_probs':
            dtype = torch.long if self.args.discrete else torch.float32
            action_true = torch.tensor(action_true, device=self.device, dtype=dtype)
        if hasattr(batch, 'teacher_action'):
            action_teacher = batch.teacher_action
        else:
            action_teacher = torch.zeros_like(action_true)
        if len(action_teacher.shape) == 2 and action_teacher.shape[1] == 1:
            action_teacher = action_teacher[:, 0]
        return obss, action_true, action_teacher

    def distill(self, batch, is_training=False, source='agent'):
        ### SETUP ###
        self.train(is_training)

        # Obtain batch and preprocess
        obss, action_true, action_teacher = self.preprocess_distill(batch, source)
        # Dropout instructions with probability instr_dropout_prob,
        # unless we have no teachers present in which case we keep the instr.
        instr_dropout_prob = 0 if self.teacher == 'none' else self.args.distill_dropout_prob
        obss = self.obs_preprocessor(obss, self.teacher, show_instrs=np.random.uniform() > instr_dropout_prob)

        ### RUN MODEL, COMPUTE LOSS ###
        _, info = self.act(obss, sample=True)
        dist = info['dist']
        if len(action_true.shape) == 3:  # Has an extra dimension
            action_true = action_true.squeeze(1)
        if len(action_true.shape) == 1: # not enough idmensions
            action_true = action_true.unsqueeze(1)
        policy_loss = -dist.log_prob(action_true).mean()
        # entropy = dist.entropy().mean()
        entropy = 0
        # TODO: consider re-adding recon loss
        loss = policy_loss  # - self.args.entropy_coef * entropy

        ### LOGGING ###
        log = self.log_distill(action_true, action_teacher, entropy, policy_loss, loss, dist, is_training)

        ### UPDATE ###
        if is_training:
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
        return log


# https://medium.com/@kengz/soft-actor-critic-for-continuous-and-discrete-actions-eeff6f651954
class GumbelSoftmax(distributions.RelaxedOneHotCategorical):
    '''
    A differentiable Categorical distribution using reparametrization trick with Gumbel-Softmax
    Explanation http://amid.fish/assets/gumbel.html
    NOTE: use this in place PyTorch's RelaxedOneHotCategorical distribution since its log_prob is not working right (returns positive values)
    Papers:
    [1] The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables (Maddison et al, 2017)
    [2] Categorical Reparametrization with Gumbel-Softmax (Jang et al, 2017)
    '''

    def sample(self, sample_shape=torch.Size(), one_hot=False):
        '''Gumbel-softmax sampling. Note rsample is inherited from RelaxedOneHotCategorical'''
        u = torch.empty(self.logits.size(), device=self.logits.device, dtype=self.logits.dtype).uniform_(0, 1)
        noisy_logits = self.logits - torch.log(-torch.log(u))
        argmax = torch.argmax(noisy_logits, dim=-1, keepdim=True)
        if one_hot:
            argmax = F.one_hot(argmax[:, 0].long(), self.logits.shape[-1]).float()
        return argmax

    def rsample(self, sample_shape=torch.Size(), one_hot=False):
        soft_sample = super().rsample(sample_shape)
        hard_sample = torch.argmax(soft_sample, dim=-1, keepdim=True)
        if one_hot:
            hard_sample = F.one_hot(hard_sample[:, 0].long(), self.logits.shape[-1]).float()
        return soft_sample, hard_sample

    def log_prob(self, value):
        '''value is one-hot or relaxed'''
        if value.shape != self.logits.shape:
            value = F.one_hot(value[:, 0].long(), self.logits.shape[-1]).float()
            assert value.shape == self.logits.shape
        return - torch.sum(- value * F.log_softmax(self.logits, -1), -1)
