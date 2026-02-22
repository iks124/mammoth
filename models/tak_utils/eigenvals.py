import torch

def fast_eigvals(A: torch.Tensor) -> torch.Tensor:
    return torch.lobpcg(A.double(), k=1, method='ortho')[0]

def normal_eigvals(A: torch.Tensor) -> torch.Tensor:
    return torch.linalg.eigvalsh(A.double())

@torch.no_grad()
def compute_eigenvalues(ggT: dict[str, list[torch.Tensor]], aaT: dict[str, list[torch.Tensor]], ffT: dict[str, torch.Tensor], fast=True) -> dict[str, torch.Tensor]:
    """
    ggT: gram products of gradients
    aaT: gram products of activations
    ffT: full fisher matrix (layer norms)
    """
    assert all([n in aaT for n in ggT]), "ggT keys must be in aaT keys"

    ggT_eigenvalues = {}
    aaT_eigenvalues = {}
    ffT_eigenvalues = {}

    for name, gg in ggT.items():
        ggT_eigenvalues[name] = fast_eigvals(gg[0]) if fast else normal_eigvals(gg[0])
        aaT_eigenvalues[name] = fast_eigvals(aaT[name][0]) if fast else normal_eigvals(aaT[name][0])

    for name, ff in ffT.items():
        # Compute eigenvalues and eigenvectors for the full fisher matrix
        ffT_eigenvalues[name] = fast_eigvals(ffT[name]) if fast else normal_eigvals(ffT[name])

    layer_eigens: dict[str, torch.Tensor] = {}
    for key in ggT_eigenvalues.keys():
        layer_eigens[key] = torch.einsum('i,j->ij',ggT_eigenvalues[key], aaT_eigenvalues[key]) # https://en.wikipedia.org/wiki/Kronecker_product - Abstract properties
    for key in ffT_eigenvalues.keys():
        layer_eigens[key] = ffT_eigenvalues[key]

    return layer_eigens

@torch.no_grad()
def compute_eigenvalues_old(ggT: dict[str, torch.Tensor], aaT: dict[str, torch.Tensor], ffT: dict[str, torch.Tensor], fast=True) -> dict[str, torch.Tensor]:
    """
    ggT: gram products of gradients
    aaT: gram products of activations
    ffT: full fisher matrix (layer norms)
    """
    assert all([n in aaT for n in ggT]), "ggT keys must be in aaT keys"

    ggT_eigenvalues = {}
    aaT_eigenvalues = {}
    ffT_eigenvalues = {}

    for name, gg in ggT.items():
        ggT_eigenvalues[name] = fast_eigvals(gg) if fast else normal_eigvals(gg)
        aaT_eigenvalues[name] = fast_eigvals(aaT[name]) if fast else normal_eigvals(aaT[name])

    for name, ff in ffT.items():
        # Compute eigenvalues and eigenvectors for the full fisher matrix
        ffT_eigenvalues[name] = fast_eigvals(ffT[name]) if fast else normal_eigvals(ffT[name])

    layer_eigens: dict[str, torch.Tensor] = {}
    for key in ggT_eigenvalues.keys():
        layer_eigens[key] = torch.einsum('i,j->ij',ggT_eigenvalues[key], aaT_eigenvalues[key]) # https://en.wikipedia.org/wiki/Kronecker_product - Abstract properties
    for key in ffT_eigenvalues.keys():
        layer_eigens[key] = ffT_eigenvalues[key]

    return layer_eigens