// Tiled fused bilinear upsampling.
// Same math as bilinear.metal, but a whole TEAM of threads first copies the
// input pixels it needs into fast on-chip "threadgroup" memory, once, and
// then every thread in that team reads its four corners from that shared
// copy instead of going back to device memory. Neighbouring output pixels
// need mostly the same input pixels, so this avoids repeating those reads.
//
// This file is only the kernel BODY (see the comment at the top of
// bilinear.metal for what MLX fills in around it: buffers, grid position,
// template constants). Extra template constants used only here:
// TW, TH   - tile size in output pixels (width, height).
// Cc       - channels handled by one team.
// Ho_pad   - H_out rounded up to a multiple of TH (see bilinear.py).

// threadgroup_position_in_grid: which TEAM this thread belongs to (3D).
// thread_position_in_threadgroup: this thread's place inside its team.
// A team's id times its shape (Cc, TW, TH) gives the FIRST output pixel the
// team is responsible for: the tile's origin.
uint tg_c = threadgroup_position_in_grid.x * Cc;
uint tg_ox = threadgroup_position_in_grid.y * TW;
uint tg_z = threadgroup_position_in_grid.z * TH;

uint lx = thread_position_in_threadgroup.x; // local channel offset, 0..Cc-1
uint ly = thread_position_in_threadgroup.y; // local column offset, 0..TW-1
uint lz = thread_position_in_threadgroup.z; // local row offset,    0..TH-1

// The grid's row axis is padded to a multiple of TH per batch image
// (Ho_pad), so a tile never covers rows from two different images.
// Unpack row and batch the same way for this thread and for the tile
// origin. tile_oy/tile_b are the same for every thread in the team,
// because TH divides Ho_pad.
uint gz = tg_z + lz;
uint oy = gz % Ho_pad;
uint b = gz / Ho_pad;
uint tile_oy = tg_z % Ho_pad;
uint tile_b = tg_z / Ho_pad;
uint c = tg_c + lx;
uint ox = tg_ox + ly;

// The padding only ever adds whole trailing teams, so a tile's origin
// (tg_c, tg_ox, tile_oy) always sits inside the real output. But some
// individual threads inside a team still land past the edge (for example
// C is not a multiple of Cc) and must not write an output value.
// They must still help load the tile below, so this check is used later,
// not as an early return now.
bool valid = (c < C) && (ox < W_out) && (oy < H_out);

// Footprint: the block of input pixels this whole tile needs.
// With scale >= 1, each output pixel's two corners per axis move forward
// by at most one input pixel as ox/oy increases by one, so a TW x TH tile
// needs at most (TW+1) x (TH+1) input pixels.
uint last_ox = min(tg_ox + (uint)TW, (uint)W_out) - 1;
uint last_oy = min(tile_oy + (uint)TH, (uint)H_out) - 1;
uint in_x0 = x_l[tg_ox];
uint in_y0 = y_l[tile_oy];
uint Fw = x_r[last_ox] - in_x0 + 1;
uint Fh = y_r[last_oy] - in_y0 + 1;
uint Fc = min((uint)Cc, (uint)C - tg_c);

// Shared buffer for the tile, sized for the worst case any team can need:
// (TH+1) rows x (TW+1) columns x Cc channels. Stored in the input dtype T.
// Laid out with fixed, compile-time strides (TW+1, Cc) so every index into
// it below is a multiply-add, never a division: tile[(row*(TW+1)+col)*Cc+ch].
// Only the sub-block row<Fh, col<Fw, ch<Fc is actually filled in.
threadgroup T tile[(TH + 1) * (TW + 1) * Cc];

// Cooperative load: every thread in the team copies some pixels into
// `tile`, striding by the team's row/column extent so the work is spread
// evenly. Each thread owns one channel (its own lx) and walks a grid of
// (row, col) positions, so there is no division or modulo by a runtime
// value anywhere in this loop (only compile-time-constant strides, TW+1 and
// Cc, are used to place each element in `tile`).
// MLX dispatches with Metal's non-uniform threadgroups: a team at the edge
// of the grid can have FEWER threads than (Cc, TW, TH), and then
// threads_per_threadgroup reports that smaller, actual size for it. That
// means at a channel edge, threads_per_threadgroup.x already equals Fc, so
// every thread that exists there already has lx < Fc; the check below is
// just making that explicit.
uint tpy = threads_per_threadgroup.y;
uint tpz = threads_per_threadgroup.z;
uint base = tile_b * H_in * W_in * C;
uint ch = lx;
if (ch < Fc) {
    uint ic = tg_c + ch;
    for (uint r = lz; r < Fh; r += tpz) {
        uint iy = in_y0 + r;
        for (uint col = ly; col < Fw; col += tpy) {
            uint ix = in_x0 + col;
            tile[(r * (TW + 1) + col) * Cc + ch] = x[base + (iy * W_in + ix) * C + ic];
        }
    }
}

// Every thread, valid or not, must reach this barrier: it is what makes
// the load above finish before anyone reads from `tile`.
threadgroup_barrier(mem_flags::mem_threadgroup);

// Only now is it safe for an out-of-range thread to stop.
if (!valid) return;

// Same corner lookup as bilinear.metal, but the indices are relative to
// the tile's input origin, and the read comes from fast threadgroup
// memory instead of device memory.
uint y0 = y_l[oy] - in_y0, y1 = y_r[oy] - in_y0;
uint x0 = x_l[ox] - in_x0, x1 = x_r[ox] - in_x0;
float ty = y_w[oy], tx = x_w[ox];
uint cr = lx; // channel offset within the tile is just the local x id

float q00 = tile[(y0 * (TW + 1) + x0) * Cc + cr];
float q10 = tile[(y0 * (TW + 1) + x1) * Cc + cr];
float q01 = tile[(y1 * (TW + 1) + x0) * Cc + cr];
float q11 = tile[(y1 * (TW + 1) + x1) * Cc + cr];

float r0 = (1 - tx) * q00 + tx * q10;
float r1 = (1 - tx) * q01 + tx * q11;

out[((b * H_out + oy) * W_out + ox) * C + c] = Tout((1 - ty) * r0 + ty * r1);
