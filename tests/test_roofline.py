import pandas as pd

from mlx_upsample.roofline import dtype_bytes, matmul_flops, min_bytes, uses_matmul


def _row(**kw):
    base = dict(B=1, H=4, W=4, C=1, scale=2.0, align_corners=False, dtype="float32", pass_="fwd")
    base.update(kw)
    pas = base.pop("pass_")
    return pd.Series({**base, "pass": pas})


def test_dtype_bytes():
    assert dtype_bytes("float16") == 2
    assert dtype_bytes("float32") == 4


def test_fwd_bwd_bytes_adds_grad_traffic():
    row = _row()
    fwd = min_bytes(row, out_elem=4)
    fwd_bwd = min_bytes(_row(pass_="fwd_bwd"), out_elem=4)
    # fwd_bwd adds one grad-output read (= output bytes) and one full grad-input write.
    n_out = 1 * 8 * 8 * 1
    n_in_full = 1 * 4 * 4 * 1
    assert fwd_bwd == fwd + n_out * 4 + n_in_full * 4


def test_out_elem_changes_output_bytes_only():
    row32 = _row(dtype="float32")
    row16 = _row(dtype="float16")
    # out_elem fixed at 4: only the input-side bytes shrink for float16.
    assert min_bytes(row32, out_elem=4) - min_bytes(row16, out_elem=4) == (1 * 4 * 4 * 1) * (4 - 2)
    # out_elem following the input dtype also shrinks the output bytes.
    assert min_bytes(row16, out_elem=2) < min_bytes(row16, out_elem=4)


def test_downscale_read_cap():
    # H=W=100, scale=0.1 -> oh=ow=10; a correct kernel reads at most 2*10=20 per axis,
    # far less than the full 100x100 input.
    row = _row(H=100, W=100, scale=0.1, pass_="fwd")
    b = min_bytes(row, out_elem=4)
    n_in_read_capped = 1 * min(100, 20) * min(100, 20) * 1
    n_out = 1 * 10 * 10 * 1
    assert b == n_in_read_capped * 4 + n_out * 4
    assert n_in_read_capped < 100 * 100  # the cap actually bites


def test_upscale_not_capped():
    # H=W=4, scale=2 -> oh=ow=8; cap min(H, 2*oh) = min(4, 16) = 4 = H, not binding.
    row = _row(H=4, W=4, scale=2.0, pass_="fwd")
    b = min_bytes(row, out_elem=4)
    n_in_full = 1 * 4 * 4 * 1
    n_out = 1 * 8 * 8 * 1
    assert b == n_in_full * 4 + n_out * 4


def test_matmul_flops_hand_computed():
    # B=1, H=W=2, C=1, scale=2 (upscale, align_corners=False) -> oh=ow=4.
    # Axis 1 (H): contract K=H=2 over "other" dims B*W*C*oh = 1*2*1*4 = 8 -> 2*2*8 = 32.
    # Axis 2 (W): contract K=W=2 over "other" dims B*oh*C*ow = 1*4*1*4 = 16 -> 2*2*16 = 64.
    row = _row(H=2, W=2, C=1, scale=2.0, pass_="fwd")
    assert matmul_flops(row) == 32 + 64 == 96


def test_matmul_flops_backward_is_double_forward():
    row_fwd = _row(H=2, W=2, C=1, scale=2.0, pass_="fwd")
    row_bwd = _row(H=2, W=2, C=1, scale=2.0, pass_="fwd_bwd")
    assert matmul_flops(row_bwd) == 2 * matmul_flops(row_fwd)


def test_uses_matmul_gate():
    assert uses_matmul(_row(H=64, W=64, scale=2.0)) is True  # upscale, small
    assert uses_matmul(_row(H=64, W=64, scale=0.5)) is False  # downscale
    assert uses_matmul(_row(H=256, W=256, scale=2.0)) is False  # upscale, too large
