import torch
import numpy as np

def compute_gradient_attention(model, input_batch, device):
    """
    Compute gradient attention map (G_dp) from the model and input batch of images.
    input_batch: [B, C, D, H, W]
    """
    input_batch = input_batch.to(device).requires_grad_(True)
    logits = model(input_batch)  # [B, num_classes]
    log_probs = torch.log_softmax(logits, dim=1)

    # Sum over all classes for a scalar loss, or choose a specific class if desired.
    loss = log_probs.sum()

    # Compute gradients
    gradient = torch.autograd.grad(outputs=loss, inputs=input_batch, create_graph=False)[0]
    return gradient

def normalize_gdp(gdp):
    gdp_min, gdp_max = gdp.min(), gdp.max()
    if gdp_min == gdp_max:
        # If by any chance min == max, just return gdp as is to avoid division by zero.
        # This is very unlikely, but good to handle gracefully.
        gradient_normalized = gdp
    else:
        gradient_normalized = (gdp - gdp_min) / (gdp_max - gdp_min)
    return gradient_normalized.detach()

def preprocess_log_jsm(log_jsm, threshold=0.2):
    """
    Preprocess the log Jacobian map to highlight the magnitude of changes.
    Positive: expansion, Negative: contraction, Zero: no change.

    Steps:
    1) Take absolute value (so expansion/contraction are treated equally by magnitude).
    2) Zero out small changes (abs < threshold).
    3) Normalize remaining values to [0, 1].
    """
    abs_dev = np.abs(log_jsm)  # Consider magnitude only

    # Step 2: Zero out small changes
    abs_dev[abs_dev < threshold] = 0.0

    # Step 3: Min-max normalization
    abs_min = np.min(abs_dev)
    abs_max = np.max(abs_dev)

    if abs_min == abs_max:
        # If everything is zero (or the same), just return abs_dev as-is
        norm = abs_dev
    else:
        # Normalize to [0, 1]
        norm = (abs_dev - abs_min) / (abs_max - abs_min)

    return norm


def preprocess_log_jsm_negative(log_jsm, threshold=0.2):
    """
    Preprocess the log Jacobian map to account for both expansion and contraction.
    Positive: expansion, Negative: contraction, Zero: no change.

    Steps:
    1) Threshold small changes (|log_jsm| < threshold => 0).
    2) Normalize the remaining values to [-1, 1], preserving the sign:
       - Positive values = expansion
       - Negative values = contraction
       - Zero = no change
    """
    # 1) Zero out values whose absolute magnitude is below the threshold
    jsm_copy = log_jsm.copy()
    jsm_copy[np.abs(jsm_copy) < threshold] = 0.0

    # 2) Compute min and max of the thresholded map
    dev_min = jsm_copy.min()
    dev_max = jsm_copy.max()

    # If everything is zero or the entire range is constant, just return as-is
    if dev_min == dev_max:
        # e.g., all zeros or same value
        norm = jsm_copy
    else:
        # 3) Scale to [-1, 1] while preserving sign
        # Formula: norm = 2*(x - min) / (max - min) - 1
        norm = 2.0 * (jsm_copy - dev_min) / (dev_max - dev_min) - 1.0

    return norm


def combine_jsm_and_gdp(log_jsm, gdp, alpha=0.7):
    """
    Combine log JSM and GDP into a final attention map.
    log_jsm: [B, 1, D, H, W] - log jacobian map
    gdp: [B, 1, D, H, W] - gradient-based attention map (Gdp)

    Steps:
    1. Preprocess log_jsm to get a [0,1]-scaled deviation map.
    2. Normalize gdp.
    3. Combine with alpha weighting.
    """
    abs_norm = preprocess_log_jsm(log_jsm)
    gdp_normalized = normalize_gdp(gdp)
    return alpha * abs_norm + (1 - alpha) * gdp_normalized

def apply_attention_map(images, attention_map):
    # images, attention_map both [B, 1, D, H, W]
    return images * attention_map

