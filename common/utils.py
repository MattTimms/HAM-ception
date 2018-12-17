import torch
import torch.nn as nn


def unnormalise(img):
    min, max = float(img.min()), float(img.max())
    image = img.clamp_(min=min, max=max)
    image = image.add_(-min).div_(max - min + 1e-5)
    image = image.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return image


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        device (torch.Device): device on which tensors will be allocated to.
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, device, sigma=0.1, is_relative_detach=True):
        super(GaussianNoise, self).__init__()
        self.device = device

        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).float().to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x
