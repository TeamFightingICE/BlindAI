import psutil
import torch


def kill_proc_tree(pid: int, including_parent=True):    
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.kill()
    gone, still_alive = psutil.wait_procs(children, timeout=5)
    if including_parent:
        parent.kill()
        parent.wait(5)


def normalize_reward(reward: float):
    return (reward + 400) / 800


def compute_advantages(rewards: torch.Tensor, values: torch.Tensor, discount: float, gae_lambda: float) -> torch.Tensor:
    """
    Compute General Advantage.
    """
    deltas = rewards + discount * values[1:] - values[:-1]
    seq_len = len(rewards)
    advs = torch.zeros(seq_len + 1)
    multiplier = discount * gae_lambda
    for i in range(seq_len - 1, -1, -1):
        advs[i] = advs[i + 1] * multiplier + deltas[i]
    return advs[:-1]


def calc_discounted_return(rewards: torch.Tensor, discount: float, final_value: float) -> torch.Tensor:
    """
    Calculate discounted returns based on rewards and discount factor.
    """
    seq_len = len(rewards)
    discounted_returns = torch.zeros(seq_len)
    discounted_returns[-1] = rewards[-1] + discount * final_value
    for i in range(seq_len - 2, -1, -1):
        discounted_returns[i] = rewards[i] + discount * discounted_returns[i + 1]
    return discounted_returns


def magic_combine(x: torch.Tensor, dim_begin: int, dim_end: int) -> torch.Tensor:
    combined_shape = list(x.shape[:dim_begin]) + [-1] + list(x.shape[dim_end:])
    return x.view(combined_shape)
