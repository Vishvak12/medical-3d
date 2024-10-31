from typing import Union, List, Tuple, Dict
from PIL import Image

import torch
from torch.nn import functional as F
from transformers.models.zoedepth.modeling_zoedepth import ZoeDepthDepthEstimatorOutput
import numpy as np

def post_process_depth_estimation_default(
    outputs, target_sizes: Union[torch.Tensor, List[Tuple[int, int]]] = None
) -> List[Dict] :
    """
    Converts the raw output of [`*DepthEstimatorOutput`] into final depth predictions and depth PIL image.
    Only supports PyTorch.

    Args:
        outputs ([`*DepthEstimatorOutput`]):
            Raw outputs of the model.
        target_sizes (`torch.Tensor` or `List[Tuple[int, int]]`, *optional*):
            Tensor of shape `(batch_size, 2)` or list of tuples (`Tuple[int, int]`) containing the target size
            (height, width) of each image in the batch. If left to None, predictions will not be resized.

    Returns:
        `List[Dict]`: A list of dictionaries, each dictionary containing the depth predictions and a depth PIL
        image as predicted by the model.
    """
    predicted_depth = outputs.predicted_depth

    if (target_sizes is not None) and (len(predicted_depth) != len(target_sizes)):
        raise ValueError(
            "Make sure that you pass in as many target sizes as the batch dimension of the predicted depth"
        )

    results = []
    for i, d in enumerate(predicted_depth):
        if target_sizes is not None:
            target_size = target_sizes[i]
            d = F.interpolate(d, size=target_size, mode="bicubic", align_corners=False)
        d = d.squeeze().cpu().numpy()
        pil = Image.fromarray((d * 255 / np.max(d)).astype("uint8"))

        results.append({"predicted_depth": d, "depth": pil})

    return results