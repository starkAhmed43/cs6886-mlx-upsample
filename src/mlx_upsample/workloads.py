"""Synthetic grid and real-model catalog of bilinear upsampling workloads."""

from dataclasses import dataclass
from itertools import product

import mlx.core as mx

CAP_BYTES = 512 * 2**20


@dataclass(frozen=True)
class Workload:
    name: str
    source: str
    set: str  # "grid" | "real"
    B: int
    H: int
    W: int
    C: int
    scale: float
    align_corners: bool
    dtype: str  # "float32" | "float16"


def out_hw(w: Workload) -> tuple[int, int]:
    return int(w.scale * w.H), int(w.scale * w.W)


def out_bytes(w: Workload) -> int:
    oh, ow = out_hw(w)
    return w.B * oh * ow * w.C * 4


def _capped(ws: list[Workload], capped: bool) -> list[Workload]:
    return [w for w in ws if out_bytes(w) <= CAP_BYTES] if capped else ws


def grid(capped: bool = True) -> list[Workload]:
    ws = [
        Workload(f"grid.h{n}.s{s:g}.c{c}", "synthetic grid", "grid", 1, n, n, c, s, False, d)
        for n, s, c, d in product(
            (32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048),
            (0.25, 0.5, 1.5, 2, 3, 4),
            (4, 32, 256),
            ("float32", "float16"),
        )
    ]
    return _capped(ws, capped)


def _scale(h: int, w: int, oh: int, ow: int) -> float:
    """Scale with int(scale * N) == target on both axes."""
    s = oh / h + 1e-9
    assert (int(s * h), int(s * w)) == (oh, ow), (h, w, oh, ow)
    return s


# (name, source, H, W, C, out H, out W, align_corners, training batch, batch basis)
_MMSEG = "mmseg 8 GPUs x 2/GPU; per-device B=2"
_CATALOG = [
    *[
        (f"rife.{n}", "RIFE, issue #4435 shapes (not config-verified)", h, w, 4, oh, ow, False, 8, "B=8 assumed, per-device")
        for n, h, w, oh, ow in [
            ("up4", 192, 320, 768, 1280),
            ("up2", 384, 640, 768, 1280),
            ("down4", 768, 1280, 192, 320),
            ("down2", 768, 1280, 384, 640),
        ]
    ],
    *[
        (f"{m}.mlp{n}", f"HF {repo} config: decoder_hidden_size={c}, 512 input, 1/4 stage 128", n, n, c, 128, 128, False, 2, _MMSEG)
        for m, repo, c in [
            ("segformer_b0", "nvidia/segformer-b0-finetuned-ade-512-512", 256),
            ("segformer_b2", "nvidia/segformer-b2-finetuned-ade-512-512", 768),
        ]
        for n in (64, 32, 16)
    ],
    ("segformer.logits", "HF segformer ade-512 config: 150 labels, logits 128 -> 512", 128, 128, 150, 512, 512, False, 2, _MMSEG),
    *[
        (f"upernet.fpn{n}", "HF openmmlab/upernet-convnext-small config: hidden_size=512, FPN top-down x2", n, n, 512, 2 * n, 2 * n, False, 2, _MMSEG)
        for n in (16, 32, 64)
    ],
    ("upernet.logits", "HF openmmlab/upernet-convnext-small config: 150 labels, logits 128 -> 512", 128, 128, 150, 512, 512, False, 2, _MMSEG),
    ("deeplabv3p.aspp", "DeepLabV3+ paper, VOC 513 (from paper, not config-verified)", 33, 33, 256, 129, 129, True, 2, _MMSEG),
    ("deeplabv3p.logits", "DeepLabV3+ paper, VOC 513, 21 classes (from paper, not config-verified)", 129, 129, 21, 513, 513, True, 2, _MMSEG),
    *[
        (f"pspnet.bin{n}", "PSPNet paper, 473 crop, bins to 60 (from paper, not config-verified)", n, n, 512, 60, 60, True, 2, _MMSEG)
        for n in (1, 2, 3, 6)
    ],
    *[
        (f"{m}.fuse{n}", f"HF LiheYoung/{repo} config: fusion_hidden_size={c}, patch 14, 518 input", n, n, c, 2 * n, 2 * n, True, 4, "B=4 assumed, per-device")
        for m, repo, c in [
            ("depth_anything_s", "depth-anything-small-hf", 64),
            ("depth_anything_l", "depth-anything-large-hf", 256),
        ]
        for n in (37, 74, 148)
    ],
    ("sam.masks", "HF facebook/sam-vit-base config: 256 low-res masks -> 1024, 3 multimask outputs", 256, 256, 3, 1024, 1024, False, 4, "B=4 assumed, per-device"),
]


def real(capped: bool = True) -> list[Workload]:
    ws = [
        Workload(name, f"{src}; {basis}", "real", b, h, w, c, _scale(h, w, oh, ow), ac, dt)
        for name, src, h, w, c, oh, ow, ac, tb, basis in _CATALOG
        for b, dt in ((1, "float16"), (tb, "float32"))
    ]
    return _capped(ws, capped)


def make_input(w: Workload, seed: int = 0) -> mx.array:
    key = mx.random.key(seed)
    return mx.random.normal((w.B, w.H, w.W, w.C), key=key).astype(getattr(mx, w.dtype))


if __name__ == "__main__":
    for label, fn in (("grid", grid), ("real", real)):
        print(f"{label}: {len(fn(capped=False))} before cap, {len(fn())} after cap")
    for w in real():
        print(f"{w.name:24s} B={w.B:<2d} {w.dtype:8s} {w.H}x{w.W}x{w.C} -> {out_hw(w)}")
