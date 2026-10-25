from mlx_upsample.workloads import CAP_BYTES, grid, make_input, out_bytes, out_hw, real


def test_grid_counts():
    assert len(grid(capped=False)) == 12 * 6 * 3 * 2
    g = grid()
    assert 0 < len(g) < 432
    assert all(out_bytes(w) <= CAP_BYTES for w in g)


def test_real_hits_intended_output_size():
    ws = real(capped=False)
    assert len(ws) % 2 == 0
    sizes = {w.name: out_hw(w) for w in ws}
    assert sizes["deeplabv3p.logits"] == (513, 513)
    assert sizes["deeplabv3p.aspp"] == (129, 129)
    assert {sizes[f"pspnet.bin{n}"] for n in (1, 2, 3, 6)} == {(60, 60)}
    assert sizes["rife.up4"] == (768, 1280) and sizes["rife.down4"] == (192, 320)
    assert sizes["sam.masks"] == (1024, 1024)


def test_make_input():
    w = real()[0]
    x = make_input(w)
    assert x.shape == (w.B, w.H, w.W, w.C) and str(x.dtype).endswith(w.dtype)
