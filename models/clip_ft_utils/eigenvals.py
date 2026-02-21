import torch


def fast_eigvals(a: torch.Tensor) -> torch.Tensor:
    return torch.lobpcg(a.double(), k=1, method="ortho")[0]


def normal_eigvals(a: torch.Tensor) -> torch.Tensor:
    return torch.linalg.eigvalsh(a.double())


@torch.no_grad()
def compute_eigenvalues(
    ggt: dict[str, list[torch.Tensor]],
    aat: dict[str, list[torch.Tensor]],
    fft: dict[str, torch.Tensor],
    fast=True,
) -> dict[str, torch.Tensor]:
    assert all(n in aat for n in ggt)
    ggt_eig = {}
    aat_eig = {}
    fft_eig = {}

    for name, gg in ggt.items():
        ggt_eig[name] = fast_eigvals(gg[0]) if fast else normal_eigvals(gg[0])
        aat_eig[name] = (
            fast_eigvals(aat[name][0]) if fast else normal_eigvals(aat[name][0])
        )

    for name, ff in fft.items():
        fft_eig[name] = fast_eigvals(ff) if fast else normal_eigvals(ff)

    layer_eigens: dict[str, torch.Tensor] = {}
    for key in ggt_eig.keys():
        layer_eigens[key] = torch.einsum("i,j->ij", ggt_eig[key], aat_eig[key])
    for key in fft_eig.keys():
        layer_eigens[key] = fft_eig[key]
    return layer_eigens
