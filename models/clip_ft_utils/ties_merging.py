import torch


def topk_values_mask(matrix, keep_ratio=0.7, return_mask=False):
    if keep_ratio > 1:
        keep_ratio /= 100

    if keep_ratio >= 1 and return_mask:
        ones = torch.ones_like(matrix)
        return matrix, ones.mean(dim=-1), ones
    if keep_ratio >= 1:
        return matrix, torch.ones_like(matrix).mean(dim=-1)

    original_shape = matrix.shape
    if matrix.dim() == 1:
        matrix = matrix.unsqueeze(0)

    _, d = matrix.shape
    k = int(d * keep_ratio)
    k = d - k
    if matrix.flatten().shape[-1] == 1:
        kth_values = matrix.abs()
    else:
        kth_values, _ = matrix.abs().kthvalue(k, dim=1, keepdim=True)

    mask = matrix.abs() >= kth_values
    if original_shape == matrix.squeeze().shape:
        final_mask = mask.squeeze()
        matrix = matrix.squeeze()
    else:
        final_mask = mask

    if return_mask:
        return matrix * final_mask, final_mask.float().mean(dim=-1), final_mask
    return matrix * final_mask, final_mask.float().mean(dim=-1)


def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())
    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult


def chunked_sum(tensor, chunk_size=10000):
    num_chunks = tensor.size(0) // chunk_size + (
        1 if tensor.size(0) % chunk_size != 0 else 0
    )
    total_sum = torch.zeros_like(tensor[0])
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, tensor.size(0))
        chunk = tensor[start_idx:end_idx]
        total_sum += torch.sum(chunk, dim=0)
    return total_sum


def disjoint_merge(tensor, merge_func, reference_sign_to_mult, weights=None):
    if reference_sign_to_mult is not None:
        rows_to_keep = torch.where(
            reference_sign_to_mult.unsqueeze(0) > 0, tensor > 0, tensor < 0
        )
    else:
        rows_to_keep = tensor != 0

    selected_entries = tensor * rows_to_keep
    if weights is not None:
        for selected_entry in selected_entries:
            selected_entry *= weights[0]

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = chunked_sum(selected_entries)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= reference_sign_to_mult
    elif merge_func == "unmerged":
        disjoint_aggs = selected_entries
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs, rows_to_keep


def resolve_sign(tensor):
    sign_to_mult = torch.sign(tensor.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult


def ties_merging(vectors, topk=20, merging_type="mean", weights=None, **kwargs):
    original_shape = vectors[0].shape
    flat_list = [v.reshape(-1) for v in vectors]
    stacked_vectors = torch.vstack(flat_list).clone()
    pruned_vectors, _, mask = topk_values_mask(
        stacked_vectors, keep_ratio=topk, return_mask=True
    )
    vector_signs = resolve_sign(pruned_vectors)
    merged_tv, rows_to_keep = disjoint_merge(
        pruned_vectors, merging_type, vector_signs, weights
    )
    merged_tv = merged_tv.reshape(original_shape)
    return merged_tv, rows_to_keep, mask
