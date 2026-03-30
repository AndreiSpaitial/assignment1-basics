import math


def cosine_annealing(
    t: int,
    lr_max: float,
    lr_min: float,
    t_W: int,
    t_C: int,
) -> float:
    if t < t_W:
        return lr_max * t/t_W
    if t <= t_C:
        return (
            lr_min +
            0.5 * (
                1 +
                math.cos(math.pi * (t-t_W)/(t_C-t_W))) *
                (lr_max - lr_min)
        )

    return lr_min
