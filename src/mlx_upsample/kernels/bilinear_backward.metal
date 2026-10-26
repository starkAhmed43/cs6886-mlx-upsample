// This file holds only the BODY of a Metal kernel, not a full kernel function.
// bilinear.py wraps it with mx.fast.metal_kernel, the same way as bilinear.metal.
// MLX writes the function signature, the input and output buffers
// (g, y_l, y_r, y_w, x_l, x_r, x_w, y_lo, y_hi, x_lo, x_hi, gx), the
// thread_position_in_grid argument, and the template constants
// (T, C, H_in, W_in, H_out, W_out).
//
// This is the gradient (backward) kernel for bilinear.metal's forward kernel.
// Forward: one thread per OUTPUT pixel, each reading 4 input pixels.
// Backward: one thread per INPUT pixel, each reading every output pixel
// that the forward pass read from it, and adding up their share of the
// incoming gradient g. Doing it this way (scatter turned into gather) means
// no two threads ever write the same output value, so there is no need for
// atomic adds, which are slow on the GPU.
//
// Each input pixel i is only read by a contiguous range of output pixels,
// because the forward tables (y_l, y_r, x_l, x_r) are non-decreasing in the
// output index. y_lo/y_hi and x_lo/x_hi (built on the CPU with numpy, see
// bilinear.py) give that range per axis: i is read by every output index o
// with y_lo[i] <= o < y_hi[i] (same idea for x). An input pixel that no
// output pixel reads (can happen when downscaling) gets lo == hi, i.e. an
// empty range, and its gradient is 0.

uint c = thread_position_in_grid.x;
uint ix = thread_position_in_grid.y;
uint iy = thread_position_in_grid.z % H_in;
uint b = thread_position_in_grid.z / H_in;

// Guard: threads outside the input must do nothing (same reasoning as the
// forward kernel's guard: cheap, and keeps this correct under any grid).
if (c >= C || ix >= W_in) return;

uint y_lo_i = y_lo[iy], y_hi_i = y_hi[iy];
uint x_lo_i = x_lo[ix], x_hi_i = x_hi[ix];

uint base = b * H_out * W_out * C;

// Accumulate in float for accuracy, even when T (and so the final write)
// is float16.
float sum = 0;
for (uint oy = y_lo_i; oy < y_hi_i; oy++) {
    // This output row's weight on iy: (1 - ty) if iy is its left neighbour,
    // ty if iy is its right neighbour. Both can apply at once at a clipped
    // edge, where y_l[oy] == y_r[oy] == iy.
    float ty = y_w[oy];
    float wy = 0;
    if (y_l[oy] == iy) wy += (1 - ty);
    if (y_r[oy] == iy) wy += ty;
    // Hoist wy out of the ox loop: it does not depend on ox.
    for (uint ox = x_lo_i; ox < x_hi_i; ox++) {
        float tx = x_w[ox];
        float wx = 0;
        if (x_l[ox] == ix) wx += (1 - tx);
        if (x_r[ox] == ix) wx += tx;
        sum += wy * wx * g[base + (oy * W_out + ox) * C + c];
    }
}

gx[((b * H_in + iy) * W_in + ix) * C + c] = T(sum);
