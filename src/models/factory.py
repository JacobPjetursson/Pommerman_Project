from .model_pomm import PommNet
from .model_generic import CNNBase, MLPBase
from .policy import Policy


def create_policy(obs_space, nn_kwargs={}, train=True):
    obs_shape = obs_space.shape
    nn = PommNet(obs_shape=obs_shape, **nn_kwargs)

    if train:
        nn.train()
    else:
        nn.eval()

    policy = Policy(nn)

    return policy
