"""get_spiral_yxs: node positions come from arithmetic, not data, so they must not drift with
winding count (villa #2027). Building angles in float32 puts winding 250 half a voxel off."""
import numpy as np
import torch

from sample_spiral import get_spiral_yxs


def _float64_reference(num_windings, dr, spacing):
    out = []
    for k in range(num_windings):
        theta = k * 2 * np.pi + np.arange(0, 2 * np.pi, spacing / (k + 0.5) / dr, dtype=np.float64)
        radius = dr * theta / (2 * np.pi)
        out.append(np.stack([np.sin(theta), np.cos(theta)], axis=-1) * radius[:, None])
    return out


def test_outer_windings_sit_on_the_float64_spiral():
    yxs = get_spiral_yxs(256, 16, 20, group_by_winding=True, device="cpu")
    ref = _float64_reference(256, 16, 20)
    assert [len(a) for a in yxs] == [len(b) for b in ref]       # same nodes, only better placed
    assert sum(len(a) for a in yxs) == 164836
    err = np.concatenate([np.linalg.norm(a.numpy().astype(np.float64) - b, axis=-1)
                          for a, b in zip(yxs, ref)])
    assert err.max() < 1e-2                                     # float32 angles gave 0.4852 at winding 250
    assert yxs[0].dtype == torch.get_default_dtype()


def test_dtype_device_and_gradient_through_dr_are_unchanged():
    dr = torch.tensor(16.0, requires_grad=True)
    yxs = get_spiral_yxs(3, dr, 4, device="cpu")
    assert yxs.dtype == torch.float32 and yxs.device == torch.device("cpu") and yxs.requires_grad
    yxs.sum().backward()
    assert dr.grad is not None and torch.isfinite(dr.grad)
