// This file holds only the BODY of a Metal kernel, not a full kernel function.
// mx.fast.metal_kernel in bilinear.py wraps it.
// MLX writes the function signature, the input and output buffers
// (x, y_l, y_r, y_w, x_l, x_r, x_w, out), the thread_position_in_grid argument,
// and the template constants (T, Tout, C, H_in, W_in, H_out, W_out).
// So this file does not compile on its own.
// An editor may flag unknown names. That is expected.
//
// This is Metal code (a C++-like language for Apple GPUs).
// The body runs once for every GPU thread, all at the same time.
// Each thread computes ONE output value: one output pixel in one channel.
// There is no loop over pixels. The grid of threads is the loop.

// thread_position_in_grid is this thread's 3D id.
// MLX sees this name in the source and adds it as a kernel argument.
// x = channel, y = output column.
// z = output row and batch packed together, because a grid has only 3 axes.
// "% H_out" and "/ H_out" unpack the row and the batch.
uint c = thread_position_in_grid.x;
uint ox = thread_position_in_grid.y;
uint oy = thread_position_in_grid.z % H_out;
uint b = thread_position_in_grid.z / H_out;

// Guard: threads outside the output must do nothing.
// Metal rounds the launch up to whole threadgroups, so it can start more
// threads than "grid" asks for (for example C=33 with threadgroup x=32).
// The extra threads land outside the output. This guard keeps them safe.
if (c >= C || ox >= W_out) return;

// For each axis: two neighbour input indices (l = left/low, r = right/high).
// The weights y_w and x_w are blend fractions t in 0..1.
// MLX precomputes them and clips indices to [0, N-1], so reads stay in bounds.
uint y0 = y_l[oy], y1 = y_r[oy], x0 = x_l[ox], x1 = x_r[ox];
float ty = y_w[oy], tx = x_w[ox];

// Arrays are flat, row-major, NHWC (channels last).
// Index of (b, y, x, c) = ((b*H + y)*W + x)*C + c.
// "base" jumps to the start of batch b.
// Neighbouring threads have consecutive c, so they read neighbouring memory.
// This is called a coalesced read, and it is fast.
uint base = b * H_in * W_in * C;

// The four corner pixels around the sample point.
// Name qXY: first digit is the x corner (0 or 1), second is the y corner.
// The math uses float for accuracy, even if the input is float16.
float q00 = x[base + (y0 * W_in + x0) * C + c];
float q10 = x[base + (y0 * W_in + x1) * C + c];
float q01 = x[base + (y1 * W_in + x0) * C + c];
float q11 = x[base + (y1 * W_in + x1) * C + c];

// Blend along x on the bottom edge (r0) and the top edge (r1).
float r0 = (1 - tx) * q00 + tx * q10;
float r1 = (1 - tx) * q01 + tx * q11;

// Blend along y, then write the result.
// Tout(...) casts to the output dtype. Stock MLX always computes in
// float32, so Tout is float32 even when the input T is float16.
// This is "fusion": one pass reads 4 values and writes 1.
// Stock MLX does 4 separate gathers and builds 4 full-size temporary arrays.
out[((b * H_out + oy) * W_out + ox) * C + c] = Tout((1 - ty) * r0 + ty * r1);
