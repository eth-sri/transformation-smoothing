from robustness.attack_steps import AttackerStep
import torch as ch


class GaussianL2Step(AttackerStep):

    """
    Imagenet Training like in Salman.
    Only works for Enable Random Start, 1 Step
    """

    # static variable for noise
    sigma = 1
    clamp = True
    vingette = None

    def __init__(self, orig_input, eps, step_size, use_grad=True):
        super().__init__(orig_input, eps, step_size, use_grad=True)
        self.noise = ch.randn_like(orig_input) * GaussianL2Step.sigma

    def project(self, x):
        if GaussianL2Step.clamp:
            x = ch.clamp(x - self.noise, 0, 1) + self.noise
        if GaussianL2Step.vingette is not None:
            x = GaussianL2Step.vingette(x)
        return x

    def step(self, x, g):
        l = len(x.shape) - 1
        g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1] * l))
        scaled_g = g / (g_norm + 1e-10)
        return x + self.eps * scaled_g

    def random_perturb(self, x):
        return self.orig_input + self.noise


class GaussianNoise(AttackerStep):

    sigma = 1
    vingette = None

    def __init__(self, orig_input, eps, step_size, use_grad=False):
        super().__init__(orig_input, eps, step_size, use_grad=False)
        self.noise = ch.randn_like(orig_input) * GaussianNoise.sigma

    def project(self, x):
        x =  self.orig_input + self.noise
        if GaussianNoise.vingette is not None:
            x = GaussianNoise.vingette(x)
        return x

    def step(self, x, g):
        return x

    def random_perturb(self, x):
        return x
