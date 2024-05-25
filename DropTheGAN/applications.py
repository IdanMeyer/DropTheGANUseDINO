import math
from typing import Tuple

import torch

import gpnn
import resize_right
# Seed 1 gives artifact in the left
# Seed 5 gives artifact in the right, going down to the water
# torch.manual_seed(6) # Top ballon is cut
# torch.manual_seed(11) # Bottom part is cut between water and sky
# torch.manual_seed(2) # river-modified - right part quite blurry
# torch.manual_seed(1) # river-modified - right part quite blurry
# torch.manual_seed(2) # river-modified - right part quite blurry
# torch.manual_seed(3) # Birds - a bit blurry
torch.manual_seed(0)

_INF = float('inf')


def generation(image: torch.Tensor,
               noise_std: float = 0.75,
               alpha: float = _INF,
               patch_size: int = 7,
               downscale_ratio: float = 0.75,
               num_levels: int = 9,
               reduce: str = 'weighted_mean') -> torch.Tensor:
    pyramid = gpnn.make_pyramid(image, num_levels, downscale_ratio)
    initial_guess = pyramid[-1] + noise_std * torch.randn_like(pyramid[-1])
    return gpnn.gpnn(pyramid,
                     initial_guess,
                     alpha=alpha,
                     downscale_ratio=downscale_ratio,
                     patch_size=patch_size,
                     reduce=reduce)


def editing(source_image: torch.Tensor,
            edited_image: torch.Tensor,
            alpha: float = _INF,
            patch_size: int = 7,
            downscale_ratio: float = 0.75,
            num_levels: int = 5,
            reduce: str = 'weighted_mean') -> torch.Tensor:
    source_pyramid = gpnn.make_pyramid(source_image, num_levels,
                                       downscale_ratio)
    edited_pyramid = gpnn.make_pyramid(edited_image, num_levels,
                                       downscale_ratio)
    initial_guess = edited_pyramid[-1]
    return gpnn.gpnn(source_pyramid,
                     initial_guess,
                     alpha=alpha,
                     downscale_ratio=downscale_ratio,
                     patch_size=patch_size,
                     reduce=reduce)


def conditional_inpainting(masked_image: torch.Tensor,
                           mask: torch.Tensor,
                           alpha: float = _INF,
                           patch_size: int = 7,
                           downscale_ratio: float = 0.75,
                           num_levels: int = 5,
                           reduce: str = 'weighted_mean') -> torch.Tensor:
    pyramid = gpnn.make_pyramid(masked_image, num_levels, downscale_ratio)
    mask_pyramid = gpnn.make_pyramid(mask.to(masked_image), num_levels,
                                     downscale_ratio)
    initial_guess = pyramid[-1]
    return gpnn.gpnn(pyramid,
                     initial_guess,
                     mask_pyramid=mask_pyramid,
                     alpha=alpha,
                     downscale_ratio=downscale_ratio,
                     patch_size=patch_size,
                     reduce=reduce)


def structural_analogy(source_image: torch.Tensor,
                       structure_image: torch.Tensor,
                       alpha: float = 5e-3,
                       patch_size: int = 7,
                       downscale_ratio: float = 0.75,
                       num_levels: int = 5,
                       reduce: str = 'weighted_mean',
                       should_use_our_code=True) -> torch.Tensor:
    source_pyramid = gpnn.make_pyramid(source_image, num_levels,
                                       downscale_ratio)
    structure_pyramid = gpnn.make_pyramid(structure_image, num_levels,
                                          downscale_ratio)
    output_pyramid_shape = [x.shape for x in structure_pyramid]
    initial_guess = structure_pyramid[-1]
    return gpnn.gpnn(source_pyramid,
                     initial_guess,
                     output_pyramid_shape=output_pyramid_shape,
                     alpha=alpha,
                     downscale_ratio=downscale_ratio,
                     patch_size=patch_size,
                     reduce=reduce,
                     should_use_our_code=should_use_our_code)


def retargeting(image: torch.Tensor,
                retargeting_ratio: Tuple[float, float],
                alpha: float = 1e-3,
                patch_size: int = 7,
                downscale_ratio: float = 0.8,
                num_levels: int = 9,
                reduce: str = 'weighted_mean'):
    pyramid = gpnn.make_pyramid(image, num_levels, downscale_ratio)
    retargeted_generated = resize_right.resize(image, retargeting_ratio)
    retargeted_pyramid = gpnn.make_pyramid(retargeted_generated, num_levels,
                                           downscale_ratio)
    retargeted_pyramid_shape = [level.shape for level in retargeted_pyramid]
    initial_quess = retargeted_pyramid[-1]
    return gpnn.gpnn(pyramid,
                     initial_quess,
                     output_pyramid_shape=retargeted_pyramid_shape,
                     alpha=alpha,
                     downscale_ratio=downscale_ratio,
                     patch_size=patch_size,
                     reduce=reduce,
                     num_iters_in_coarsest_level=10)
