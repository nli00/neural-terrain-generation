import torch

def mask_inputs(
    indices: torch.Tensor,
    mask_token_id: int,
    mask_scheduler,
    ignore_index = -100
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized operation to mask proportion of each sequence using masking scheduler.

    Args:
        indices: (B, L) integer codebook indices in [0, codebook_size - 1].
        mask_token_id: integer id written at masked positions (expected: codebook_size).
        mask_scheduler: scheduler; input (B,) in (0, 1], output (B,) mask fractions.
    """
    if indices.dim() != 2:
        raise ValueError(f"indices must be (B, L); got shape {tuple(indices.shape)}")

    device = indices.device
    B, L = indices.shape

    random_iters = torch.rand(B, device=device) # These inputs to the mask scheduler are drawn from a uniform (0,1] and correspond to random iterations in the masking schedule used in decoding
    mask_ratios = mask_scheduler(random_iters)

    num_to_mask = (mask_ratios * L).floor().long().clamp(1, L - 1)

    noise = torch.rand(B, L, device = device)

    sorted_noise, _ = torch.sort(noise, dim = 1) # Sort the noise along the sequence axis
    thresholds = torch.gather(sorted_noise, dim = 1, index = num_to_mask.view(-1, 1) - 1) # need view -1, 1 because num_to_mask has shape B and sorted noise has shape B, L. This view makes them both 2D and alignes the B dim
    
    mask = noise <= thresholds # <= here because if thresholds grabs the value of the 5th item, and we want 5 items, we should include that item.
    
    masked_indices = indices.clone()
    masked_indices[mask] = mask_token_id

    labels = torch.full_like(indices, ignore_index)
    labels[mask] = indices[mask]

    return masked_indices, labels

def outpaint_right(
    indices: torch.Tensor, # B x L
    mask_token_id: int,
    mask_proportion: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Test horizontal generation with a sliding window to the right.

    Assumes that each set of indices in the batch are arranged in a square (1D input).
    Rolls the rows to the left and masks the right n columns.

    Returns:
        masked_flat: Masked and rolled indices, shape (B, L)
        mask_flat: Boolean mask, 1 for original/unmasked, 0 for masked_token_id positions, shape (B, L)
    """
    assert (mask_proportion > 0) and (mask_proportion < 1)
    B = indices.shape[0]
    L = indices.shape[1]
    side = int(L ** 0.5)
    assert side * side == L, "Length of indices must be a perfect square."

    indices_square = indices.view(B, side, side)
    n = max(1, int(mask_proportion * side))

    # Roll each row n to the left
    rolled_indices = torch.roll(indices_square, shifts=-n, dims=2)

    # Mask rightmost n columns
    masked_indices = rolled_indices.clone()
    masked_indices[:, :, -n:] = mask_token_id

    # Flatten back to 1D
    masked_flat = masked_indices.view(B, -1)

    # Boolean mask: 1 for unedited/original index, 0 for masked
    boolean_mask = (masked_flat != mask_token_id)  # Will be 1 for unmasked, 0 for masked

    return masked_flat, boolean_mask

