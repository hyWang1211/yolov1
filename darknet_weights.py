import struct

import numpy as np
import torch
from torch import nn

EXTRACTION_CONV_COUNT = 20
EXTRACTION_CLASSES = 1000


@torch.no_grad()
def load_extraction_weights(model, path):
    """Initialize the detector backbone with the published Darknet Extraction model."""
    layers = [layer for layer in model.features if isinstance(layer, nn.Conv2d)]
    layers = layers[:EXTRACTION_CONV_COUNT]
    if len(layers) != EXTRACTION_CONV_COUNT:
        raise ValueError("Extraction requires 20 matching convolutional layers.")

    with path.open("rb") as stream:
        header = stream.read(16)
        if len(header) != 16 or struct.unpack("<3i", header[:12]) != (0, 1, 0):
            raise ValueError("Expected the published Darknet extraction.weights header (version 0.1.0).")

        conv_values = sum(layer.weight.numel() + 4 * layer.out_channels for layer in layers)
        classifier_values = EXTRACTION_CLASSES * (layers[-1].out_channels + 1)
        expected_bytes = len(header) + 4 * (conv_values + classifier_values)
        if stream.seek(0, 2) != expected_bytes:
            raise ValueError(f"Expected a complete extraction.weights file of {expected_bytes:,} bytes.")
        stream.seek(len(header))

        for index, layer in enumerate(layers, start=1):
            channels = layer.out_channels
            count = 4 * channels + layer.weight.numel()
            values = np.fromfile(stream, dtype="<f4", count=count)
            if values.size != count or not np.isfinite(values).all():
                raise ValueError(f"Invalid weights in Extraction convolution {index}.")
            stats = torch.from_numpy(values[:4 * channels]).view(4, channels)
            bias, scale, mean, variance = stats.unbind(0)
            if (variance < 0).any():
                raise ValueError(f"Invalid BatchNorm variance in Extraction convolution {index}.")
            kernel = torch.from_numpy(values[4 * channels:]).view_as(layer.weight)
            # Darknet folds inference-time BatchNorm using its CUDA epsilon.
            factor = scale * torch.rsqrt(variance + 1e-5)
            layer.weight.copy_(kernel * factor.view(-1, 1, 1, 1))
            layer.bias.copy_(bias - mean * factor)
    return len(layers)
