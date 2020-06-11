import argparse

import torch
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--algo', default='adam', help='algorithm to use: adam | rmsprop | sgd | tdprop')
    parser.add_argument(
        '--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--beta-1',
        type=float,
        default=0.0,
        help='optimizer beta_1 (default: 0.0)')
    parser.add_argument(
        '--beta-2',
        type=float,
        default=0.99,
        help='optimizer beta2 (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=0.5,
        help='max norm of gradients (default: 0.5)')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--num-processes',
        type=int,
        default=16,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--num-steps',
        type=int,
        default=5,
        help='number of forward steps in A2C (default: 5)')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='BreakoutNoFrameskip-v4',
        help='environment to train on (default: BreakoutNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='./logs/',
        help='directory to save agent logs (default: ./logs/)')
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')
    parser.add_argument(
        '--exploration',
        type=float,
        default=0.01,
        help='policy epsilon')
    parser.add_argument(
        '--array_name', default='', help='(optional) hyperparameter exploration array name')
    parser.add_argument(
        '--array_id', type=int, default=0, help='(optional) hyperparameter exploration array job id')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.array_name:
        apply_array(eval(args.array_name)(), args.array_id, args)
    assert args.algo in ['tdprop', 'adam', 'rmsprop', 'sgd']

    return args

def apply_array(params, job_id, args):
    for k, v in params[job_id].items():
        setattr(args, k, v)

def Jun6_10M():
    rng = np.random.RandomState(142857)
    num_runs = 50
    sgd_lr = 10**(rng.uniform(-4, 0, 1000))
    lr = 10**(rng.uniform(-5, -2, 1000))
    eps = 10**(rng.uniform(-8, -1, 1000))
    beta_2 = rng.uniform(0, 1, 1000)
    return [
        {
            # Fixed:
            'num_env_steps': 10_000_000,
            'beta_1': 0,
            'env_name': f'{env_name}NoFrameskip-v4',
            # Variable:
            'lr': lr[run] if algo != 'sgd' else sgd_lr[run],
            'eps': eps[run],
            'beta_2': beta_2[run],
            'log_dir': f'./logs/Jun6_10M_{algo}_{env_name}_{run}/',
            'algo': algo,
            'seed': run + 1,
        }
        for run in range(num_runs)
        for algo in ['sgd', 'adam', 'tdprop']
        for env_name in ['BeamRider', 'SpaceInvaders', 'Breakout', 'Qbert']]

if __name__ == "__main__":
    import sys
    for i,n in enumerate(eval(sys.argv[-1])()):
        print(i,n)
