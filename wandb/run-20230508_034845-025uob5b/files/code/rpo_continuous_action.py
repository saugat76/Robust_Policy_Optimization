# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/rpo/#rpo_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import wandb


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ml__project__rpo",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Pendulum-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=2000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--rpo-alpha", type=float, default=0.5,
        help="the alpha parameter for RPO")
    parser.add_argument("--num-nodes", type=int, default=64,
        help="number of nodes used for actor and critic n/w")
    parser.add_argument("--activation-func", type=str, default="Tanh",
        help="activation function used in actor and critic n/w: 'ReLu', 'Tanh', 'Sigmoid'")
    parser.add_argument("--optimizer-choice", type=str, default="Adam",
        help="optimizer function used for actor and critic n/w: 'Adam', 'AdamW', 'RMSProp'")
    parser.add_argument("--num-layers", type=str, default=3,
        help="number of layers for actor and critic network // only for hyperparamter record")
    
    
    
    # Addition by Saugat
    parser.add_argument("--sweep-config", type=lambda x: bool(strtobool(x)), default=False,
        help="use the sweep config provided in code // wandb instead of passed sys args")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# Defination of a single game agent 
class Agent(nn.Module):
    def __init__(self, envs, rpo_alpha, num_nodes, activation_func):
        super().__init__()
        self.rpo_alpha = rpo_alpha
        # Changed by Saugat
        # Setting of the nodes used in actor and critic network nodes as a sys args variable
        # Since the functions are not serializable so, using if else to select the respective activation function
        if activation_func  == "ReLU":
            activation_fun_layer = nn.ReLU
        elif activation_func  == "Tanh":
            activation_fun_layer = nn.Tanh
        elif activation_func == "Sigmoid":
            activation_fun_layer = nn.Sigmoid 


        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), num_nodes)),
            activation_fun_layer(),
            layer_init(nn.Linear(num_nodes, num_nodes)),
            activation_fun_layer(),
            layer_init(nn.Linear(num_nodes, num_nodes)),
            activation_fun_layer(),
            layer_init(nn.Linear(num_nodes, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), num_nodes)),
            activation_fun_layer(),
            layer_init(nn.Linear(num_nodes, num_nodes)),
            activation_fun_layer(),
            layer_init(nn.Linear(num_nodes, num_nodes)),
            activation_fun_layer(),
            layer_init(nn.Linear(num_nodes, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    # Use of epsilon greedy policy to select random action or select action based on the RPO algorithm 
    # Exxploration vs Exploitation 
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        else:  
            # New to RPO
            # Sample to add the stochasticity to the system 
            z = torch.FloatTensor(action_mean.shape).uniform_(-self.rpo_alpha, self.rpo_alpha)
            action_mean = action_mean.cuda() + z.cuda()
            probs = Normal(action_mean, action_std)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# Addition of sweep configutation using weights and biases to compute for different values
# Out optimization goal for the hyperparameter sweep is to maximize the episodic reward 
# Addition by Saugat
args = parse_args()

# sweep_configuration = {
#     'method': 'bayes',
#     'name': 'sweep',
#     'metric': {
#         'goal': 'maximize', 
#         'name': 'episodic_return'
#     },

#     'parameters': {
#         'lr': {'values': [3e-4, 1e-1, 2.5e-2, 2.5e-3, 1e-4]},
#         'num_nodes': {'values': [64, 128, 256]},
#         'optimizer_choices': {'values': ["Adam", "RMSProp", "AdamW"]}, 
#         'activation_funcs': {'values': ["ReLU", "Tanh", "Sigmoid"]}
#     }
# }

# sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.exp_name)
# Based on these sweep parameters changing the codes 

device = torch.device("cuda:1" if torch.cuda.is_available() and args.cuda else "cpu")
# device = "cpu"

def main():
    # Keep track of all the runs and store all the log information in wandb and tensorboard logs
    # May use wandb to do hyperparameter tunning and evaluate the model performance 
    run_name = f"3layers__{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        run = wandb.init(
             project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
        # Added by Saugat 
        # learning_rate = float(wandb.config.lr)
        # num_nodes = int(wandb.config.num_nodes)
        # optimizer_choice = str(wandb.config.optimizer_choices)
        # activation_func = str(wandb.config.activation_funcs)
    # else:
    #     learning_rate = args.learning_rate
    #     num_nodes = args.num_nodes
    #     optimizer_choice = "Adam"
    #     activation_func = "Tanh"    

    # Params  initilaized from sys arg without change in remaining code
    learning_rate = args.learning_rate
    num_nodes = args.num_nodes
    optimizer_choice = args.optimizer_choice
    activation_func = args.activation_func
        
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding of the all the randomness asssociated with the system variable 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    
    # Added by Saugat 
    # Since the functions are not serializable so, using if else to select the respective activation function
    agent = Agent(envs, args.rpo_alpha, num_nodes, activation_func).to(device)

    # Added by Saugat 
    # Change the optimizer function based on the sweep config // default use Adam optimizer
    if optimizer_choice == "Adam":
        optimizer = optim.Adam(agent.parameters(), lr=learning_rate, eps=1e-5)
    elif optimizer_choice == "RMSProp":
        optimizer = optim.RMSprop(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    elif optimizer_choice == "AdamW":    
        optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    else:
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)


    # ALGO Logic: Storage setup 
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Starting of the episode
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    video_filenames = set()

    for update in range(1, num_updates + 1):
        # Annealing the learning rate if instructed to do so
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execution of the game environment and logging of all the important data 
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # Only print when at least 1 env is done
            if "final_info" not in infos:
                continue

            for info in infos["final_info"]:
                # Skip the envs that are not done
                if info is None:
                    continue
                # Added by Saugat 
                # npArray is unhashable so, converting to float to do sweep
                episodic_return = float(info['episode']['r'])
                episodic_length = float(info["episode"]["l"])
                print(f"global_step={global_step}, episodic_return: {episodic_return}")
                writer.add_scalar("charts/episodic_return", episodic_return, global_step)
                writer.add_scalar("charts/episodic_length", episodic_length, global_step)

                # Addition of code to keep track of the convergence related paramters into weights and biases to sweep of hyperparamters
                # Addition by Saugat 
                if args.track:
                    wandb.log({"episodic_return": episodic_return, 
                               "episodic_length": episodic_length, 
                               "global_steps": global_step})

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Log all relevent information to tensorboard
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        # wandb code to track of entropy // added by Saugat
        if args.track:
            wandb.log({"entropy":float(entropy_loss), "global_step": global_step})

        if args.track and args.capture_video:
            for filename in os.listdir(f"videos/{run_name}"):
                if filename not in video_filenames and filename.endswith(".mp4"):
                    wandb.log({f"videos": wandb.Video(f"videos/{run_name}/{filename}")})
                    video_filenames.add(filename)

    envs.close()
    wandb.finish()
    writer.close()


## Addition by Saugat for implementation in notebook 
# Sweep the code // main function usign the sweep configuration 
# For the implemntation // running in CLI ->   wandb agent sweep_id
# wandb.agent(sweep_id=sweep_id, function=main)

# Continuing without any sweep
main()