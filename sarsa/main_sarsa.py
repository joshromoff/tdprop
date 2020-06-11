import copy
import glob
import os
import time
import csv
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import algo, utils
from arguments import get_args
from envs import make_vec_envs
from our_q_model import Policy
from storage import RolloutStorage


def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    utils.cleanup_log_dir(log_dir)

    with open(log_dir + 'extras.csv', "w") as file:
        file.write("n, value_loss\n")

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    model = Policy(
        envs.observation_space.shape,
        envs.action_space.n,
        extra_kwargs={'use_backpack': args.algo == 'tdprop'})
    model.to(device)

    if args.algo == 'tdprop':
        from algo.sarsa_tdprop import SARSA
        agent = SARSA(
                model,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm,
                beta_1=args.beta_1,
                beta_2=args.beta_2,
                n=args.num_steps,
                num_processes=args.num_processes,
                gamma=args.gamma)
    else:
        from algo.sarsa import SARSA
        agent = SARSA(
            model,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            beta_1=args.beta_1,
            beta_2=args.beta_2,
            algo=args.algo)

    explore_policy = utils.eps_greedy 
    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              model.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                qs = model(rollouts.obs[step])
                _, dist = explore_policy(qs, args.exploration)
                actions = dist.sample().unsqueeze(-1)
                value = qs.gather(-1, actions)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(actions)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, torch.FloatTensor([0.0]), actions,
                            value, value, reward, masks, bad_masks)
        with torch.no_grad():
            next_qs = model(rollouts.obs[-1])
            next_probs, _ = explore_policy(next_qs, args.exploration)
            next_value = (next_probs * next_qs).sum(-1).unsqueeze(-1)

        rollouts.compute_returns(next_value, args.gamma)

        value_loss = agent.update(rollouts, explore_policy, args.exploration)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0 or j == num_updates - 1):
            save_path = os.path.join(args.log_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            torch.save([
                list(model.parameters()),
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)],
                os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                    ("Updates {}, num timesteps {}, FPS {}\n" + \
                            "Last {} training episodes: mean/median reward {:.1f}/{:.1f}" + \
                            ", min/max reward {:.1f}/{:.1f}\n" + \
                            "entropy {:.2f}, value loss {:.4f}")
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist.entropy().mean().item(), value_loss))
            with open(log_dir + 'extras.csv', "a") as file:
                file.write(str(total_num_steps) + ", " + str(value_loss)  + "\n")


if __name__ == "__main__":
    main()
