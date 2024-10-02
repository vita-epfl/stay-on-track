import numpy as np
import torch

class SmoothTurn:
    def __init__(self, attack_power, pow, border):
        """
        Initializes a SmoothTurn transformation which makes a simple turn in the road
        """
        self.b = pow
        self.a = attack_power / (self.b * 10 ** self.b)
        self.border = border

    def f(self, x):
        realx, x = x, x[x >= self.border] - self.border
        ret = torch.zeros_like(x)
        ret[x < 10] = self.a * x[x < 10] ** self.b
        ret[x >= 10] = 10 ** (self.b - 1) * self.a * self.b * x[x >= 10] - (self.b - 1) * self.a * 10 ** self.b
        realret = torch.zeros_like(realx)
        realret[realx >= self.border] = ret
        return realret

    def f_prime(self, x):
        realx, x = x, x[x >= self.border] - self.border
        ret = torch.zeros_like(x)
        ret[x < 10] = self.a * self.b * x[x < 10] ** (self.b - 1)
        ret[x >= 10] = (self.b - 1) * self.a * self.b
        realret = torch.zeros_like(realx)
        realret[realx >= self.border] = ret
        return realret

    def f_zegond(self, x):
        realx, x = x, x[x >= self.border] - self.border
        ret = torch.zeros_like(x)
        ret[x < 10] = self.a * self.b * (self.b - 1) * x[x < 10] ** (self.b - 2)
        realret = torch.zeros_like(realx)
        realret[realx >= self.border] = ret
        return realret


class DoubleTurn:
    def __init__(self, attack_power, pow, d, border):
        """
        Initializes a Double Turn transformation which is to consecutive SmoothTurns in opposite directions
        """
        self.smooth_func = SmoothTurn(attack_power, pow, border)
        self.d = d

    def f(self, x):
        return self.smooth_func.f(x) - self.smooth_func.f(x - self.d)

    def f_prime(self, x):
        return self.smooth_func.f_prime(x) - self.smooth_func.f_prime(x - self.d)

    def f_zegond(self, x):
        return self.smooth_func.f_zegond(x) - self.smooth_func.f_zegond(x - self.d)


class RippleRoad:
    def __init__(self, attack_power, l, border):
        """
        Initializes a RippleRoad transformation which makes a ripple road
        """
        self.attack_power = attack_power
        self.l = l
        self.border = border

    def f(self, x):
        realx, x = x, x[x >= self.border] - self.border
        ret = self.attack_power * (1 - torch.cos(2 * np.pi * x / self.l))
        realret = torch.zeros_like(realx)
        realret[realx >= self.border] = ret
        return realret

    def f_prime(self, x):
        realx, x = x, x[x >= self.border] - self.border
        ret = self.attack_power * 2 * np.pi / self.l * torch.sin(2 * np.pi * x / self.l)
        realret = torch.zeros_like(realx)
        realret[realx >= self.border] = ret
        return realret

    def f_zegond(self, x):
        realx, x = x, x[x >= self.border] - self.border
        ret = self.attack_power * 4 * np.pi**2 / self.l**2 * torch.cos(2 * np.pi * x / self.l)
        realret = torch.zeros_like(realx)
        realret[realx >= self.border] = ret
        return realret


class Combination:
    def __init__(self, params):
        """
        Combines the three types of the above transformations together
        :param params: a dictionary containing parameters for each transformation
        """
        self.smooth_turn = SmoothTurn(params["smooth-turn"]["attack_power"], params["smooth-turn"]["pow"], params["smooth-turn"]["border"])
        self.double_turn = DoubleTurn(params["double-turn"]["attack_power"], params["double-turn"]["pow"], params["double-turn"]["l"], params["double-turn"]["border"])
        self.ripple_road = RippleRoad(params["ripple-road"]["attack_power"], params["ripple-road"]["l"], params["ripple-road"]["border"])

    def f(self, x):
        return self.smooth_turn.f(x) + self.double_turn.f(x) + self.ripple_road.f(x)

    def f_prime(self, x):
        return self.smooth_turn.f_prime(x) + self.double_turn.f_prime(x) + self.ripple_road.f_prime(x)

    def f_zegond(self, x):
        return self.smooth_turn.f_zegond(x) + self.double_turn.f_zegond(x) + self.ripple_road.f_zegond(x)

def calc_curvature(attack_function, x):
    """
    given any set of points x, outputs the curvature of the attack_function on those points
    """
    numerator = attack_function.f_zegond(x)
    denominator = (1 + attack_function.f_prime(x) ** 2) ** 1.5
    return numerator / denominator


def calc_radius(attack_function, x):
    """
    given any set of points x, outputs the radius of a circle fitting to the attack_function on those points
    """
    curv = calc_curvature(attack_function, x)
    ret = np.zeros_like(x)
    ret[curv == 0] = 1000_000_000_000  # inf
    ret[curv != 0] = 1 / np.abs(curv[curv != 0])
    return ret


def correct_history(attack_params, history):  # inputs history points in the agents coordinate system and corrects its speed
    # calculating the minimum r of the attacking turn
    attack_function = Combination(attack_params)
    border1 = attack_params["smooth-turn"]["border"]
    border2 = attack_params["double-turn"]["border"]
    border3 = attack_params["ripple-road"]["border"]
    l_range = min(border1, border2, border3)
    r_range = max(border1 + 10, border2 + 20 + attack_params["double-turn"]["l"], border3 + attack_params["ripple-road"]["l"])

    search_points = np.linspace(l_range, r_range, 100)
    search_point_rs = calc_radius(attack_function, search_points)
    min_r = search_point_rs.min()
    g = 9.8
    miu_s = 0.7
    max_speed = np.sqrt(miu_s * g * min_r)
    current_speed = np.sqrt(((history[-1] - history[-2]) ** 2).sum()) * 10
    if current_speed <= max_speed:
        return history
    scale_factor = max_speed / current_speed
    return history * scale_factor

def apply_transform_function(attack_params, points):
    """
    Applies attack_function on the input points
    :param points: np array of points that we want to apply the transformation on
    :return: transformed points
    """
    attack_function = Combination(attack_params)
    points[..., 1] += attack_function.f(points[..., 0])
    # return points

def apply_inverse_transform_function(attack_params, points):
    """
    Applies the inverse of the transformation_function on the input points
    :param points: np array of points that we want to apply inverse transformation on
    :return: inverse transformed points
    """
    attack_function = Combination(attack_params)
    points[..., 1] -= attack_function.f(points[..., 0])

def attack_batch(batch, attack_params):
    """
    Applies the attack on the input batch
    :param batch: a batch of trajectories
    :return: the attacked batch
    """
    # obj_trajs
    apply_transform_function(attack_params, batch["input_dict"]["obj_trajs"])

    # map_polylines
    apply_transform_function(attack_params, batch["input_dict"]["map_polylines"][..., :2])
    apply_transform_function(attack_params, batch["input_dict"]["map_polylines"][..., 3:5])
    apply_transform_function(attack_params, batch["input_dict"]["map_polylines"][..., 6:8])

    # boundary_polygon_pts
    apply_transform_function(attack_params, batch["input_dict"]["boundary_polygon_pts"])
