import torch
import torch.nn as nn

# import the skrl components to build the RL system
from amp import AMP, AMP_DEFAULT_CONFIG
# from ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import yaml
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# seed for reproducibility
set_seed(56)  # e.g. `set_seed(42)` for fixed seed

class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        

        self.net = nn.Sequential(nn.Linear(self.num_observations, 1024),
                                 nn.ELU(),
                                 nn.Linear(1024, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 256),
                                 nn.ELU()
                                 )

        self.mean_layer = nn.Linear(256, self.num_actions)
        self.log_std_parameter = nn.Parameter(0.0*torch.ones(self.num_actions))


    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        
    def compute(self, inputs, role):
        if role == "policy":
            output = self.net(inputs["states"])
            return self.mean_layer(output), self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 1024),
                                 nn.ELU(),
                                 nn.Linear(1024, 512),
                                 nn.ELU(),
                                 nn.Linear(512, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 256),
                                 nn.ELU()
                                 )

        self.value_layer = nn.Linear(256, 1)

    def act(self, inputs, role):
        if role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "value":
            output = self.net(inputs["states"]) 
            return self.value_layer(output), {}
        
class Discriminator(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, amp_observation_space, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(amp_observation_space, 1024),
                                 nn.ELU(),
                                 nn.Linear(1024, 512),
                                 nn.ELU()
                                 )

        self.logits_layer = nn.Linear(512, 1)

    def act(self, inputs, role):
        if role == "discriminator":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "discriminator":
            output = self.net(inputs["states"])
            return self.logits_layer(output), {}

try:
    config_path = os.path.join(CURRENT_DIR, "config", "AMPCONFIG.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
except FileNotFoundError:
    print(f"错误：配置文件未找到在 {config_path}。请检查路径和文件是否存在。")
    exit(1)
except yaml.YAMLError as e:
    print(f"错误：配置文件YAML格式不正确。详细信息：{e}")
    exit(1)

# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name=config["task_name"])
env = wrap_env(env)

device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=config["memory_size"], num_envs=env.num_envs, device=device)
reply_buffer = RandomMemory(memory_size=config["memory_size"], num_envs=env.num_envs, device=device)
motion_dataset = RandomMemory(memory_size=config["memory_size"], num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device)
models["value"] = Critic(env.observation_space, env.action_space, device)
models["discriminator"] = Discriminator(env.observation_space, env.action_space, device=device, amp_observation_space=config["amp_observation_space"])

# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
# cfg = PPO_DEFAULT_CONFIG.copy()
cfg = AMP_DEFAULT_CONFIG.copy()
cfg["rollouts"] = config["memory_size"]  # memory_size
cfg["learning_epochs"] = config["learning_epochs"]  # learning_epochs
cfg["mini_batches"] = config["batch_size"]  # batch_size
cfg["discount_factor"] = config["discount_factor"]  # dicount_factor
cfg["lambda"] = config["lambda"]  # lambda
cfg["learning_rate"] = config["learning_rate"]  # learning_rate
cfg["learning_rate_scheduler"] = KLAdaptiveLR
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": config["learning_rate_kl_threshold"]}  # learning_rate_kl_threshold
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 0.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
# cfg["entropy_loss_scale"] = 0.1
cfg["value_loss_scale"] = 5.0
# cfg["kl_threshold"] = 0.0
# cfg["rewards_shaper"] = None
# cfg["time_limit_bootstrap"] = False
# cfg["discriminator_loss_scale"] = 2.5
# cfg["amp_batch_size"] = 512
# cfg["discriminator_batch_size"] = 4096
# cfg["discriminator_reward_scale"] = 2.0
# cfg["discriminator_logit_regularization_scale"] = 0.05
cfg["discriminator_gradient_penalty_scale"] = config["discriminator_gradient_penalty_scale"]  
# cfg["discriminator_weight_decay_scale"] = 1.0e-4
cfg["task_reward_weight"] = config["task_reward_weight"]            # task-reward weight (wG)
cfg["style_reward_weight"] = config["style_reward_weight"]  # style-reward weight (wI)
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = config["experiment"]["write_interval"]  # write_interval
cfg["experiment"]["checkpoint_interval"] = config["experiment"]["checkpoint_interval"]  # checkpoint_interval
cfg["experiment"]["directory"] = config["experiment"]["directory"]  # directory
cfg["experiment"]["wandb"] = config["experiment"]["wandb"]  # use Weights & Biases (https://wandb.ai/site)
cfg["experiment"]["wandb_kwargs"] = {
    "project": config["experiment"]["wandb_kwargs"]["project"],   # wandb_project
    "name": config["experiment"]["wandb_kwargs"]["name"]  # wandb_name
} 

agent = AMP(models=models,
            memory=memory,  # only required during training
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            amp_observation_space=config["amp_observation_space"],

            motion_dataset=motion_dataset,
            reply_buffer=reply_buffer,
            )

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": config["time_steps"], "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
if config["train"]:
    trainer.train()
else:
    # path = os.path.join(CURRENT_DIR, "amp.pt")
    path = config["checkpoint_path"]
    agent.load(path)

    # # # start evaluation
    trainer.eval()
