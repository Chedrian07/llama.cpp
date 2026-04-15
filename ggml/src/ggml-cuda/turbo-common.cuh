#pragma once

#include "common.cuh"
#include "../ggml-common-turbo.h"

// TurboQuant CUDA dequantize helpers
// These functions dequantize a full 128-element TurboQuant block to float.
// The dequantize process: unpack codebook indices -> lookup centroids -> inverse FWHT -> scale by norm

#define TURBO_DIM_CUDA  128
#define TURBO_SEED_CUDA 42

// Lloyd-Max codebook centroids (must match ggml-common.h GGML_TABLE definitions)
// These are now defined via GGML_TABLE_BEGIN in ggml-common.h with __device__ in CUDA mode.
// Use the canonical names directly: TURBO2_CENTROIDS, TURBO3_CENTROIDS, TURBO4_CENTROIDS
#define TURBO2_CENTROIDS_CUDA TURBO2_CENTROIDS
#define TURBO3_CENTROIDS_CUDA TURBO3_CENTROIDS
#define TURBO4_CENTROIDS_CUDA TURBO4_CENTROIDS

// In-place FWHT on a float array of size n (must be power of 2)
static __device__ __forceinline__ void turbo_fwht_cuda(float * x, const int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j]       = u + v;
                x[i + j + len] = u - v;
            }
        }
    }
}

// Deterministic sign flip
static __device__ __forceinline__ void turbo_sign_flip_cuda(float * x, const int n) {
    for (int i = 0; i < n; i++) {
        uint32_t h = TURBO_SEED_CUDA * 2654435761u + (uint32_t)i * 2246822519u;
        if (h >> 31) { x[i] = -x[i]; }
    }
}

// Inverse rotation: FWHT, scale by norm/n, undo sign flip
static __device__ __forceinline__ void turbo_inverse_rotate_cuda(float * buf, const int n, const float norm) {
    turbo_fwht_cuda(buf, n);
    const float scale = norm / (float)n;
    for (int i = 0; i < n; i++) {
        buf[i] *= scale;
    }
    turbo_sign_flip_cuda(buf, n);
}

// ==================== Full block dequantize functions ====================
// These write 128 floats to dst.

static __device__ __forceinline__ void turbo_dequantize_block_turbo2(const block_turbo2_0 * blk, float * dst) {
    for (int j = 0; j < TURBO_DIM_CUDA; j++) {
        int idx = (blk->qs[j / 4] >> (2 * (j % 4))) & 0x3;
        dst[j] = TURBO2_CENTROIDS_CUDA[idx];
    }
    turbo_inverse_rotate_cuda(dst, TURBO_DIM_CUDA, blk->norm);
}

static __device__ __forceinline__ void turbo_dequantize_block_turbo2h(const block_turbo2h_0 * blk, float * dst) {
    // First 32 channels: 3-bit from qs_hi (12 bytes)
    for (int j = 0; j < 32; j++) {
        int bit_offset = j * 3;
        int byte_idx = bit_offset / 8;
        int bit_idx  = bit_offset % 8;
        int val = (blk->qs_hi[byte_idx] >> bit_idx);
        if (bit_idx > 5) {
            val |= (blk->qs_hi[byte_idx + 1] << (8 - bit_idx));
        }
        val &= 0x7;
        dst[j] = TURBO3_CENTROIDS_CUDA[val];
    }
    // Remaining 96 channels: 2-bit from qs_lo (24 bytes)
    for (int j = 0; j < 96; j++) {
        int idx = (blk->qs_lo[j / 4] >> (2 * (j % 4))) & 0x3;
        dst[32 + j] = TURBO2_CENTROIDS_CUDA[idx];
    }
    turbo_inverse_rotate_cuda(dst, TURBO_DIM_CUDA, blk->norm);
}

static __device__ __forceinline__ void turbo_dequantize_block_turbo3(const block_turbo3_0 * blk, float * dst) {
    for (int j = 0; j < TURBO_DIM_CUDA; j++) {
        int bit_offset = j * 3;
        int byte_idx = bit_offset / 8;
        int bit_idx  = bit_offset % 8;
        int val = (blk->qs[byte_idx] >> bit_idx);
        if (bit_idx > 5) {
            val |= (blk->qs[byte_idx + 1] << (8 - bit_idx));
        }
        val &= 0x7;
        dst[j] = TURBO3_CENTROIDS_CUDA[val];
    }
    turbo_inverse_rotate_cuda(dst, TURBO_DIM_CUDA, blk->norm);
}

static __device__ __forceinline__ void turbo_dequantize_block_turbo3h(const block_turbo3h_0 * blk, float * dst) {
    // First 64 channels: 4-bit from qs_hi (32 bytes)
    for (int j = 0; j < 64; j++) {
        int idx = (blk->qs_hi[j / 2] >> (4 * (j % 2))) & 0xF;
        dst[j] = TURBO4_CENTROIDS_CUDA[idx];
    }
    // Remaining 64 channels: 3-bit from qs_lo (24 bytes)
    for (int j = 0; j < 64; j++) {
        int bit_offset = j * 3;
        int byte_idx = bit_offset / 8;
        int bit_idx  = bit_offset % 8;
        int val = (blk->qs_lo[byte_idx] >> bit_idx);
        if (bit_idx > 5) {
            val |= (blk->qs_lo[byte_idx + 1] << (8 - bit_idx));
        }
        val &= 0x7;
        dst[j + 64] = TURBO3_CENTROIDS_CUDA[val];
    }
    turbo_inverse_rotate_cuda(dst, TURBO_DIM_CUDA, blk->norm);
}

static __device__ __forceinline__ void turbo_dequantize_block_turbo4(const block_turbo4_0 * blk, float * dst) {
    for (int j = 0; j < TURBO_DIM_CUDA; j++) {
        int idx = (blk->qs[j / 2] >> (4 * (j % 2))) & 0xF;
        dst[j] = TURBO4_CENTROIDS_CUDA[idx];
    }
    turbo_inverse_rotate_cuda(dst, TURBO_DIM_CUDA, blk->norm);
}

// Prod types: dequantize MSE part only (QJL correction is for inner product, not element-wise)
static __device__ __forceinline__ void turbo_dequantize_block_turbop3(const block_turbop3_0 * blk, float * dst) {
    // 2-bit MSE
    for (int j = 0; j < TURBO_DIM_CUDA; j++) {
        int idx = (blk->qs[j / 4] >> (2 * (j % 4))) & 0x3;
        dst[j] = TURBO2_CENTROIDS_CUDA[idx];
    }
    turbo_inverse_rotate_cuda(dst, TURBO_DIM_CUDA, blk->norm);
}

static __device__ __forceinline__ void turbo_dequantize_block_turbop4(const block_turbop4_0 * blk, float * dst) {
    // 3-bit MSE
    for (int j = 0; j < TURBO_DIM_CUDA; j++) {
        int bit_offset = j * 3;
        int byte_idx = bit_offset / 8;
        int bit_idx  = bit_offset % 8;
        int val = (blk->qs[byte_idx] >> bit_idx);
        if (bit_idx > 5) {
            val |= (blk->qs[byte_idx + 1] << (8 - bit_idx));
        }
        val &= 0x7;
        dst[j] = TURBO3_CENTROIDS_CUDA[val];
    }
    turbo_inverse_rotate_cuda(dst, TURBO_DIM_CUDA, blk->norm);
}

static __device__ __forceinline__ void turbo_dequantize_block_turbop5(const block_turbop5_0 * blk, float * dst) {
    // 4-bit MSE
    for (int j = 0; j < TURBO_DIM_CUDA; j++) {
        int idx = (blk->qs[j / 2] >> (4 * (j % 2))) & 0xF;
        dst[j] = TURBO4_CENTROIDS_CUDA[idx];
    }
    turbo_inverse_rotate_cuda(dst, TURBO_DIM_CUDA, blk->norm);
}

// ==================== Generic dequantize dispatcher ====================

template <ggml_type type>
static __device__ __forceinline__ void turbo_dequantize_block(const void * blk, float * dst) {
    if constexpr (type == GGML_TYPE_TURBO2_0) {
        turbo_dequantize_block_turbo2((const block_turbo2_0 *)blk, dst);
    } else if constexpr (type == GGML_TYPE_TURBO2H_0) {
        turbo_dequantize_block_turbo2h((const block_turbo2h_0 *)blk, dst);
    } else if constexpr (type == GGML_TYPE_TURBO3_0) {
        turbo_dequantize_block_turbo3((const block_turbo3_0 *)blk, dst);
    } else if constexpr (type == GGML_TYPE_TURBO3H_0) {
        turbo_dequantize_block_turbo3h((const block_turbo3h_0 *)blk, dst);
    } else if constexpr (type == GGML_TYPE_TURBO4_0) {
        turbo_dequantize_block_turbo4((const block_turbo4_0 *)blk, dst);
    } else if constexpr (type == GGML_TYPE_TURBOP3_0) {
        turbo_dequantize_block_turbop3((const block_turbop3_0 *)blk, dst);
    } else if constexpr (type == GGML_TYPE_TURBOP4_0) {
        turbo_dequantize_block_turbop4((const block_turbop4_0 *)blk, dst);
    } else if constexpr (type == GGML_TYPE_TURBOP5_0) {
        turbo_dequantize_block_turbop5((const block_turbop5_0 *)blk, dst);
    } else {
        static_assert(type == -1, "unsupported turbo type");
    }
}

// Helper to check if a type is a TurboQuant type
static __device__ __host__ __forceinline__ bool is_turbo_type(ggml_type type) {
    return type >= GGML_TYPE_TURBO2_0 && type <= GGML_TYPE_TURBOP5_0;
}

// ==================== Cooperative (warp-level) block dequantize ====================
// Each of 32 warp lanes handles 4 of the 128 elements.
// Layout: lane `l` handles indices `l*4 .. l*4 + 3`.
// These functions write exactly 128 floats into `dst` (shared memory),
// producing identical bytes to the CPU sequential dequantize.
//
// Call signature:
//   cooperative_dequantize_<type>(block_ptr, dst_shared, lane_id);
// The caller MUST do a __syncwarp() after the call before reading the result.

// One FWHT butterfly stage at compile-time `len`.
// 64 butterfly pairs per stage, each of 32 lanes handles 2.
//
// The buffer lives in __shared__ memory but arrives as a generic float*.
// CUDA 12.8 targeting sm_120 miscompiles generic-pointer shared-memory
// indexing (emitting byte-stride instead of float-stride loads).  Using
// volatile forces scalar load/store at the exact computed address, which
// the compiler handles correctly.
template <int len>
static __device__ __forceinline__ void cooperative_fwht_stage_128(float * buf, const int lane) {
    volatile float * vbuf = buf;
    #pragma unroll
    for (int p_step = 0; p_step < 2; ++p_step) {
        const int pair = lane + p_step * 32;
        const int base = (pair / len) * (len * 2) + (pair % len);
        const float u  = vbuf[base];
        const float v  = vbuf[base + len];
        vbuf[base]       = u + v;
        vbuf[base + len] = u - v;
    }
    __syncwarp();
}

// Cooperative FWHT on a 128-float shared memory buffer.
// 7 explicitly-unrolled stages: len = 1, 2, 4, 8, 16, 32, 64.
static __device__ __forceinline__ void cooperative_fwht_128(float * buf, const int lane) {
    cooperative_fwht_stage_128<1 >(buf, lane);
    cooperative_fwht_stage_128<2 >(buf, lane);
    cooperative_fwht_stage_128<4 >(buf, lane);
    cooperative_fwht_stage_128<8 >(buf, lane);
    cooperative_fwht_stage_128<16>(buf, lane);
    cooperative_fwht_stage_128<32>(buf, lane);
    cooperative_fwht_stage_128<64>(buf, lane);
}

// Cooperative inverse rotation: FWHT, scale by norm/128, sign flip.
static __device__ __forceinline__ void cooperative_inverse_rotate(float * buf, const int lane, const float norm) {
    cooperative_fwht_128(buf, lane);
    volatile float * vbuf = buf;
    const float scale = norm * (1.0f / 128.0f);
    // Each lane handles 4 elements + sign flip
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i = lane * 4 + k;
        uint32_t h = TURBO_SEED_CUDA * 2654435761u + (uint32_t)i * 2246822519u;
        float val = vbuf[i] * scale;
        if (h >> 31) { val = -val; }
        vbuf[i] = val;
    }
    __syncwarp();
}

// ---- Per-type cooperative unpack functions ----
//
// All functions take a raw const char* to avoid struct-pointer casts that let
// nvcc (especially CUDA 12.8 targeting sm_120) emit widened loads whose
// alignment requirements exceed non-power-of-2 turbo block strides.
// Float fields (norm, r_norm) are read with memcpy; byte arrays are accessed
// through direct char indexing.  Both are alignment-agnostic.

// Helper: read a float from an arbitrary address without alignment assumptions.
static __device__ __forceinline__ float turbo_read_f32(const char * p) {
    float v;
    memcpy(&v, p, sizeof(float));
    return v;
}

// Byte-offset constants for each block layout (match ggml-common-turbo.h structs).
// Algorithm 1 MSE: [float norm][uint8_t qs[...]]
#define TURBO_OFF_NORM 0
#define TURBO_OFF_QS   4  /* sizeof(float) */
// Algorithm 1 fractional: [float norm][uint8_t qs_hi[...]][uint8_t qs_lo[...]]
#define TURBO2H_OFF_QS_HI  4
#define TURBO2H_OFF_QS_LO  (4 + 12)   /* 16 */
#define TURBO3H_OFF_QS_HI  4
#define TURBO3H_OFF_QS_LO  (4 + 32)   /* 36 */
// Algorithm 2 prod: [float norm][float r_norm][uint8_t qs[...]][uint8_t qjl[16]]
#define TURBOP_OFF_NORM   0
#define TURBOP_OFF_RNORM  4
#define TURBOP_OFF_QS     8  /* sizeof(float)*2 */

static __device__ __forceinline__ void cooperative_dequantize_turbo2(
    const char * __restrict__ raw, float * dst, const int lane) {
    volatile float * vdst = dst;
    const uint8_t byte = (uint8_t)raw[TURBO_OFF_QS + lane];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int idx = (byte >> (2 * k)) & 0x3;
        vdst[lane * 4 + k] = TURBO2_CENTROIDS[idx];
    }
    __syncwarp();
    cooperative_inverse_rotate(dst, lane, turbo_read_f32(raw + TURBO_OFF_NORM));
}

static __device__ __forceinline__ void cooperative_dequantize_turbo2h(
    const char * __restrict__ raw, float * dst, const int lane) {
    volatile float * vdst = dst;
    if (lane < 8) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int j = lane * 4 + k;
            const int bit_offset = j * 3;
            const int byte_idx   = bit_offset / 8;
            const int bit_idx    = bit_offset % 8;
            int val = ((uint8_t)raw[TURBO2H_OFF_QS_HI + byte_idx] >> bit_idx);
            if (bit_idx > 5) {
                val |= ((uint8_t)raw[TURBO2H_OFF_QS_HI + byte_idx + 1] << (8 - bit_idx));
            }
            val &= 0x7;
            vdst[j] = TURBO3_CENTROIDS[val];
        }
    } else {
        const int j_base  = lane * 4;
        const int lo_base = j_base - 32;
        const uint8_t byte = (uint8_t)raw[TURBO2H_OFF_QS_LO + lo_base / 4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int idx = (byte >> (2 * k)) & 0x3;
            vdst[j_base + k] = TURBO2_CENTROIDS[idx];
        }
    }
    __syncwarp();
    cooperative_inverse_rotate(dst, lane, turbo_read_f32(raw + TURBO_OFF_NORM));
}

static __device__ __forceinline__ void cooperative_dequantize_turbo3(
    const char * __restrict__ raw, float * dst, const int lane) {
    volatile float * vdst = dst;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int j = lane * 4 + k;
        const int bit_offset = j * 3;
        const int byte_idx   = bit_offset / 8;
        const int bit_idx    = bit_offset % 8;
        int val = ((uint8_t)raw[TURBO_OFF_QS + byte_idx] >> bit_idx);
        if (bit_idx > 5) {
            val |= ((uint8_t)raw[TURBO_OFF_QS + byte_idx + 1] << (8 - bit_idx));
        }
        val &= 0x7;
        vdst[j] = TURBO3_CENTROIDS[val];
    }
    __syncwarp();
    cooperative_inverse_rotate(dst, lane, turbo_read_f32(raw + TURBO_OFF_NORM));
}

static __device__ __forceinline__ void cooperative_dequantize_turbo3h(
    const char * __restrict__ raw, float * dst, const int lane) {
    volatile float * vdst = dst;
    if (lane < 16) {
        const int j_base = lane * 4;
        const uint8_t b0 = (uint8_t)raw[TURBO3H_OFF_QS_HI + j_base / 2];
        const uint8_t b1 = (uint8_t)raw[TURBO3H_OFF_QS_HI + j_base / 2 + 1];
        vdst[j_base + 0] = TURBO4_CENTROIDS[b0 & 0xF];
        vdst[j_base + 1] = TURBO4_CENTROIDS[(b0 >> 4) & 0xF];
        vdst[j_base + 2] = TURBO4_CENTROIDS[b1 & 0xF];
        vdst[j_base + 3] = TURBO4_CENTROIDS[(b1 >> 4) & 0xF];
    } else {
        const int j_global_base = lane * 4;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int jg = j_global_base + k;
            const int j_local = jg - 64;
            const int bit_offset = j_local * 3;
            const int byte_idx   = bit_offset / 8;
            const int bit_idx    = bit_offset % 8;
            int val = ((uint8_t)raw[TURBO3H_OFF_QS_LO + byte_idx] >> bit_idx);
            if (bit_idx > 5) {
                val |= ((uint8_t)raw[TURBO3H_OFF_QS_LO + byte_idx + 1] << (8 - bit_idx));
            }
            val &= 0x7;
            vdst[jg] = TURBO3_CENTROIDS[val];
        }
    }
    __syncwarp();
    cooperative_inverse_rotate(dst, lane, turbo_read_f32(raw + TURBO_OFF_NORM));
}

static __device__ __forceinline__ void cooperative_dequantize_turbo4(
    const char * __restrict__ raw, float * dst, const int lane) {
    volatile float * vdst = dst;
    const uint8_t b0 = (uint8_t)raw[TURBO_OFF_QS + lane * 2];
    const uint8_t b1 = (uint8_t)raw[TURBO_OFF_QS + lane * 2 + 1];
    vdst[lane * 4 + 0] = TURBO4_CENTROIDS[b0 & 0xF];
    vdst[lane * 4 + 1] = TURBO4_CENTROIDS[(b0 >> 4) & 0xF];
    vdst[lane * 4 + 2] = TURBO4_CENTROIDS[b1 & 0xF];
    vdst[lane * 4 + 3] = TURBO4_CENTROIDS[(b1 >> 4) & 0xF];
    __syncwarp();
    cooperative_inverse_rotate(dst, lane, turbo_read_f32(raw + TURBO_OFF_NORM));
}

// Prod types: dequantize MSE part only (QJL correction is for inner product).
// Matches the existing sequential turbo_dequantize_block_turbop{3,4,5} which also
// ignore QJL. This is correct for the KQ dot product path in fattn-vec.

static __device__ __forceinline__ void cooperative_dequantize_turbop3(
    const char * __restrict__ raw, float * dst, const int lane) {
    volatile float * vdst = dst;
    const uint8_t byte = (uint8_t)raw[TURBOP_OFF_QS + lane];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int idx = (byte >> (2 * k)) & 0x3;
        vdst[lane * 4 + k] = TURBO2_CENTROIDS[idx];
    }
    __syncwarp();
    cooperative_inverse_rotate(dst, lane, turbo_read_f32(raw + TURBOP_OFF_NORM));
}

static __device__ __forceinline__ void cooperative_dequantize_turbop4(
    const char * __restrict__ raw, float * dst, const int lane) {
    volatile float * vdst = dst;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int j = lane * 4 + k;
        const int bit_offset = j * 3;
        const int byte_idx   = bit_offset / 8;
        const int bit_idx    = bit_offset % 8;
        int val = ((uint8_t)raw[TURBOP_OFF_QS + byte_idx] >> bit_idx);
        if (bit_idx > 5) {
            val |= ((uint8_t)raw[TURBOP_OFF_QS + byte_idx + 1] << (8 - bit_idx));
        }
        val &= 0x7;
        vdst[j] = TURBO3_CENTROIDS[val];
    }
    __syncwarp();
    cooperative_inverse_rotate(dst, lane, turbo_read_f32(raw + TURBOP_OFF_NORM));
}

static __device__ __forceinline__ void cooperative_dequantize_turbop5(
    const char * __restrict__ raw, float * dst, const int lane) {
    volatile float * vdst = dst;
    const uint8_t b0 = (uint8_t)raw[TURBOP_OFF_QS + lane * 2];
    const uint8_t b1 = (uint8_t)raw[TURBOP_OFF_QS + lane * 2 + 1];
    vdst[lane * 4 + 0] = TURBO4_CENTROIDS[b0 & 0xF];
    vdst[lane * 4 + 1] = TURBO4_CENTROIDS[(b0 >> 4) & 0xF];
    vdst[lane * 4 + 2] = TURBO4_CENTROIDS[b1 & 0xF];
    vdst[lane * 4 + 3] = TURBO4_CENTROIDS[(b1 >> 4) & 0xF];
    __syncwarp();
    cooperative_inverse_rotate(dst, lane, turbo_read_f32(raw + TURBOP_OFF_NORM));
}

// ==================== Algorithm 2 (prod) cooperative QJL helpers ====================
//
// These are used ONLY by the flash attention vec_dot path (not by the generic
// to_float dequant path) because the QJL correction is for inner products,
// not element-wise reconstruction.
//
// The QJL correction estimator (per coord, in the rotated basis) is:
//     r_hat_rotated[i] ~= (||r||/sqrt(d)) * proj_sign_i * (2*qjl_bit_i - 1)
// and the contribution to <q, k> is:
//     ||k|| * <q, R^-1(r_hat_rotated)>
//   = (||k|| * ||r|| / sqrt(d)) * <q, R^-1(sign_vec)>
// where sign_vec[i] = proj_sign_i * (2*qjl_bit_i - 1).
//
// We compute R^-1(sign_vec) cooperatively into a shared buffer (one float per
// coordinate) using the same FWHT + sign-flip primitives that are used for the
// MSE part. The "norm" passed to the inverse rotation is 1.0f because the
// (||k|| * ||r|| / sqrt(d)) scale is applied at the very end of vec_dot.

#define TURBO_QJL_SEED_VECDOT 137u

// Per-coordinate deterministic projection sign (matches CPU/CUDA quantize encoders).
// Returns +1 or -1.
static __device__ __forceinline__ float turbo_qjl_proj_sign(const int i) {
    const uint32_t h = TURBO_QJL_SEED_VECDOT * 2654435761u + (uint32_t)i * 2246822519u;
    return (h & 0x80000000u) ? -1.0f : 1.0f;
}

// Cooperative inverse rotation but with norm=1.0 (no extra scaling).
// This is the same as cooperative_inverse_rotate(buf, lane, 1.0f) but spelled
// out to allow the compiler to fold the constant scale.
static __device__ __forceinline__ void cooperative_inverse_rotate_unit(float * buf, const int lane) {
    cooperative_fwht_128(buf, lane);
    volatile float * vbuf = buf;
    constexpr float scale = 1.0f / 128.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i = lane * 4 + k;
        const uint32_t h = TURBO_SEED_CUDA * 2654435761u + (uint32_t)i * 2246822519u;
        float val = vbuf[i] * scale;
        if (h >> 31) { val = -val; }
        vbuf[i] = val;
    }
    __syncwarp();
}

// Cooperative compute of R^-1(sign_vec) for a turbop block, written to dst[128].
// Each lane handles 4 coordinates: [lane*4 .. lane*4 + 3].
//
// dst[i] = R^-1(proj_sign[i] * (2*qjl_bit[i] - 1))
//
// The result is unscaled; the caller must multiply by (||k|| * ||r|| / sqrt(D))
// when adding to the dot product.
static __device__ __forceinline__ void cooperative_qjl_signs_to_orig(
    const uint8_t * __restrict__ qjl, float * dst, const int lane) {
    volatile float * vdst = dst;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int i  = lane * 4 + k;
        const int bit = (qjl[i / 8] >> (i % 8)) & 1;
        const float ps = turbo_qjl_proj_sign(i);
        // sign_vec[i] = proj_sign_i * (2*bit - 1)
        vdst[i] = ps * (bit ? 1.0f : -1.0f);
    }
    __syncwarp();
    cooperative_inverse_rotate_unit(dst, lane);
}

// Generic cooperative dispatcher — takes raw const char*, no struct-pointer casts.
template <ggml_type type>
static __device__ __forceinline__ void cooperative_dequantize_block(
    const void * blk, float * dst, const int lane) {
    const char * raw = (const char *)blk;
    if constexpr (type == GGML_TYPE_TURBO2_0) {
        cooperative_dequantize_turbo2(raw, dst, lane);
    } else if constexpr (type == GGML_TYPE_TURBO2H_0) {
        cooperative_dequantize_turbo2h(raw, dst, lane);
    } else if constexpr (type == GGML_TYPE_TURBO3_0) {
        cooperative_dequantize_turbo3(raw, dst, lane);
    } else if constexpr (type == GGML_TYPE_TURBO3H_0) {
        cooperative_dequantize_turbo3h(raw, dst, lane);
    } else if constexpr (type == GGML_TYPE_TURBO4_0) {
        cooperative_dequantize_turbo4(raw, dst, lane);
    } else if constexpr (type == GGML_TYPE_TURBOP3_0) {
        cooperative_dequantize_turbop3(raw, dst, lane);
    } else if constexpr (type == GGML_TYPE_TURBOP4_0) {
        cooperative_dequantize_turbop4(raw, dst, lane);
    } else if constexpr (type == GGML_TYPE_TURBOP5_0) {
        cooperative_dequantize_turbop5(raw, dst, lane);
    } else {
        static_assert(type == -1, "unsupported turbo type");
    }
}
