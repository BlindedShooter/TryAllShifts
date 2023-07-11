import torch

# from "Revisiting the Minimalist Approach to Offline Reinforcement Learning" (https://arxiv.org/pdf/2006.15134.pdf)


def hopper_is_done_func(next_obs):
    if len(next_obs.shape) == 1:
        next_obs = next_obs.unsqueeze(0)

    height = next_obs[:, 0]
    angle = next_obs[:, 1]
    not_done = torch.isfinite(next_obs).all(axis=-1) \
                * (torch.abs(next_obs[:,1:]) < 100).all(axis=-1) \
                * (height > .7) \
                * (torch.abs(angle) < .2)

    done = ~not_done
    return done

def walker2d_is_done_func(next_obs):

    if len(next_obs.shape) == 1:
        next_obs.unsqueeze(0)


    height = next_obs[:, 0]
    ang = next_obs[:, 1]
    done = ~((height > 0.8) & (height < 2.0) &
            (ang > -1.0) & (ang < 1.0))
    
    return done

def ant_is_done_func(next_obs):

    if len(next_obs.shape) == 1:
        next_obs.unsqueeze(0)

    height = next_obs[:, 0]
    not_done = torch.isfinite(next_obs).all(axis=-1) \
                * (height >= 0.2) \
                * (height <= 1.0)

    done = ~not_done
    return done


def halfcheetah_is_done_func(next_obs):
    if len(next_obs.shape) == 1:
        next_obs.unsqueeze(0)

    done = ~torch.isfinite(next_obs).all(axis=-1)
    return done


def not_done_func(next_obs):
    return torch.zeros(next_obs.shape[0], dtype=torch.bool)


DONE_FUNCS = {
    'hopper': hopper_is_done_func,
    'walker2d': walker2d_is_done_func,
    'ant': ant_is_done_func,
    'halfcheetah': halfcheetah_is_done_func,
}