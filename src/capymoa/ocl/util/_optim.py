from collections import defaultdict

from torch.optim import Optimizer

def reset_optimizer_state(optimizer: Optimizer):
    """Resets the momentum buffers of the optimizer."""
    optimizer.__setstate__({'state': defaultdict(dict)})