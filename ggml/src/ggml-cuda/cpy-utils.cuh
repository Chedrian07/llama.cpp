#pragma once

#include "ggml-common.h"
#include "../ggml-common-turbo.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// =====================================================================
// TurboQuant device-side quantize functions
// Each function processes one block of 128 elements sequentially.
// =====================================================================

#define TURBO_SEED_QUANT 42

static __device__ __forceinline__ void turbo_fwht_dev(float * x, int n) {
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

static __device__ __forceinline__ void turbo_sign_flip_dev(float * x, int n, uint32_t seed) {
    for (int i = 0; i < n; i++) {
        uint32_t h = seed * 2654435761u + (uint32_t)i * 2246822519u;
        if (h >> 31) { x[i] = -x[i]; }
    }
}

// Forward rotate: normalize, sign flip, FWHT. Returns L2 norm.
static __device__ float turbo_forward_rotate_dev(const float * __restrict__ src, float * buf, int n) {
    float norm = 0.0f;
    for (int i = 0; i < n; i++) {
        norm += src[i] * src[i];
    }
    norm = sqrtf(norm);

    if (norm < 1e-30f) {
        for (int i = 0; i < n; i++) buf[i] = 0.0f;
        return 0.0f;
    }

    float inv_norm = 1.0f / norm;
    for (int i = 0; i < n; i++) {
        buf[i] = src[i] * inv_norm;
    }

    turbo_sign_flip_dev(buf, n, TURBO_SEED_QUANT);
    turbo_fwht_dev(buf, n);

    return norm;
}

// Nearest centroid finders
static __device__ __forceinline__ int turbo_nearest_2bit_dev(float val) {
    if (val < TURBO2_BOUNDARIES[0]) return 0;
    if (val < TURBO2_BOUNDARIES[1]) return 1;
    if (val < TURBO2_BOUNDARIES[2]) return 2;
    return 3;
}

static __device__ __forceinline__ int turbo_nearest_3bit_dev(float val) {
    if (val < TURBO3_BOUNDARIES[3]) {
        if (val < TURBO3_BOUNDARIES[1]) {
            return val < TURBO3_BOUNDARIES[0] ? 0 : 1;
        }
        return val < TURBO3_BOUNDARIES[2] ? 2 : 3;
    }
    if (val < TURBO3_BOUNDARIES[5]) {
        return val < TURBO3_BOUNDARIES[4] ? 4 : 5;
    }
    return val < TURBO3_BOUNDARIES[6] ? 6 : 7;
}

static __device__ __forceinline__ int turbo_nearest_4bit_dev(float val) {
    if (val < TURBO4_BOUNDARIES[7]) {
        if (val < TURBO4_BOUNDARIES[3]) {
            if (val < TURBO4_BOUNDARIES[1]) {
                return val < TURBO4_BOUNDARIES[0] ? 0 : 1;
            }
            return val < TURBO4_BOUNDARIES[2] ? 2 : 3;
        }
        if (val < TURBO4_BOUNDARIES[5]) {
            return val < TURBO4_BOUNDARIES[4] ? 4 : 5;
        }
        return val < TURBO4_BOUNDARIES[6] ? 6 : 7;
    }
    if (val < TURBO4_BOUNDARIES[11]) {
        if (val < TURBO4_BOUNDARIES[9]) {
            return val < TURBO4_BOUNDARIES[8] ? 8 : 9;
        }
        return val < TURBO4_BOUNDARIES[10] ? 10 : 11;
    }
    if (val < TURBO4_BOUNDARIES[13]) {
        return val < TURBO4_BOUNDARIES[12] ? 12 : 13;
    }
    return val < TURBO4_BOUNDARIES[14] ? 14 : 15;
}

// QJL encode residual (for prod types).
// Each coordinate is multiplied by a deterministic per-coord +/-1 projection sign
// before taking the sign bit. This must match the CPU encoder in ggml-quants.c
// (TURBO_QJL_SEED = 137) so the dequant/vec_dot path can reproduce the same
// projection signs at inference time.
#define TURBO_QJL_SEED_DEV 137u
static __device__ void turbo_qjl_encode_dev(const float * residual, uint8_t * qjl, int n) {
    for (int i = 0; i < n / 8; i++) qjl[i] = 0;
    for (int i = 0; i < n; i++) {
        uint32_t h = TURBO_QJL_SEED_DEV * 2654435761u + (uint32_t)i * 2246822519u;
        float proj_sign = (h & 0x80000000u) ? -1.0f : 1.0f;
        float projected = residual[i] * proj_sign;
        if (projected > 0.0f) {
            qjl[i / 8] |= (1u << (i % 8));
        }
    }
}

// --- turbo2_0 quantize ---
static __device__ void quantize_f32_turbo2_0_block(const float * __restrict__ x, block_turbo2_0 * __restrict__ y) {
    float buf[QK_TURBO];
    y->norm = turbo_forward_rotate_dev(x, buf, QK_TURBO);

    for (int i = 0; i < QK_TURBO / 4; i++) y->qs[i] = 0;
    for (int j = 0; j < QK_TURBO; j++) {
        int idx = turbo_nearest_2bit_dev(buf[j]);
        y->qs[j / 4] |= (uint8_t)(idx << (2 * (j % 4)));
    }
}

// --- turbo2h_0 quantize ---
static __device__ void quantize_f32_turbo2h_0_block(const float * __restrict__ x, block_turbo2h_0 * __restrict__ y) {
    float buf[QK_TURBO];
    y->norm = turbo_forward_rotate_dev(x, buf, QK_TURBO);

    for (int i = 0; i < 12; i++) y->qs_hi[i] = 0;
    for (int j = 0; j < 32; j++) {
        int idx = turbo_nearest_3bit_dev(buf[j]);
        int bit_offset = j * 3;
        int byte_idx = bit_offset / 8;
        int bit_idx  = bit_offset % 8;
        y->qs_hi[byte_idx] |= (uint8_t)((idx << bit_idx) & 0xFF);
        if (bit_idx > 5) {
            y->qs_hi[byte_idx + 1] |= (uint8_t)(idx >> (8 - bit_idx));
        }
    }

    for (int i = 0; i < 24; i++) y->qs_lo[i] = 0;
    for (int j = 0; j < 96; j++) {
        int idx = turbo_nearest_2bit_dev(buf[32 + j]);
        y->qs_lo[j / 4] |= (uint8_t)(idx << (2 * (j % 4)));
    }
}

// --- turbo3_0 quantize ---
static __device__ void quantize_f32_turbo3_0_block(const float * __restrict__ x, block_turbo3_0 * __restrict__ y) {
    float buf[QK_TURBO];
    y->norm = turbo_forward_rotate_dev(x, buf, QK_TURBO);

    for (int i = 0; i < 48; i++) y->qs[i] = 0;
    for (int j = 0; j < QK_TURBO; j++) {
        int idx = turbo_nearest_3bit_dev(buf[j]);
        int bit_offset = j * 3;
        int byte_idx = bit_offset / 8;
        int bit_idx  = bit_offset % 8;
        y->qs[byte_idx] |= (uint8_t)((idx << bit_idx) & 0xFF);
        if (bit_idx > 5) {
            y->qs[byte_idx + 1] |= (uint8_t)(idx >> (8 - bit_idx));
        }
    }
}

// --- turbo3h_0 quantize ---
static __device__ void quantize_f32_turbo3h_0_block(const float * __restrict__ x, block_turbo3h_0 * __restrict__ y) {
    float buf[QK_TURBO];
    y->norm = turbo_forward_rotate_dev(x, buf, QK_TURBO);

    for (int i = 0; i < 32; i++) y->qs_hi[i] = 0;
    for (int j = 0; j < 64; j++) {
        int idx = turbo_nearest_4bit_dev(buf[j]);
        y->qs_hi[j / 2] |= (uint8_t)(idx << (4 * (j % 2)));
    }

    for (int i = 0; i < 24; i++) y->qs_lo[i] = 0;
    for (int j = 0; j < 64; j++) {
        int idx = turbo_nearest_3bit_dev(buf[64 + j]);
        int bit_offset = j * 3;
        int byte_idx = bit_offset / 8;
        int bit_idx  = bit_offset % 8;
        y->qs_lo[byte_idx] |= (uint8_t)((idx << bit_idx) & 0xFF);
        if (bit_idx > 5) {
            y->qs_lo[byte_idx + 1] |= (uint8_t)(idx >> (8 - bit_idx));
        }
    }
}

// --- turbo4_0 quantize ---
static __device__ void quantize_f32_turbo4_0_block(const float * __restrict__ x, block_turbo4_0 * __restrict__ y) {
    float buf[QK_TURBO];
    y->norm = turbo_forward_rotate_dev(x, buf, QK_TURBO);

    for (int i = 0; i < QK_TURBO / 2; i++) y->qs[i] = 0;
    for (int j = 0; j < QK_TURBO; j++) {
        int idx = turbo_nearest_4bit_dev(buf[j]);
        y->qs[j / 2] |= (uint8_t)(idx << (4 * (j % 2)));
    }
}

// --- turbop3_0 quantize (2-bit MSE + 1-bit QJL) ---
static __device__ void quantize_f32_turbop3_0_block(const float * __restrict__ x, block_turbop3_0 * __restrict__ y) {
    float buf[QK_TURBO];
    float residual[QK_TURBO];
    y->norm = turbo_forward_rotate_dev(x, buf, QK_TURBO);

    for (int i = 0; i < QK_TURBO / 4; i++) y->qs[i] = 0;
    for (int j = 0; j < QK_TURBO; j++) {
        int idx = turbo_nearest_2bit_dev(buf[j]);
        y->qs[j / 4] |= (uint8_t)(idx << (2 * (j % 4)));
        residual[j] = buf[j] - TURBO2_CENTROIDS[idx];
    }

    // Residual L2 norm in rotated basis (needed for QJL correction in vec_dot)
    float r_norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO; j++) {
        r_norm_sq += residual[j] * residual[j];
    }
    y->r_norm = sqrtf(r_norm_sq);

    turbo_qjl_encode_dev(residual, y->qjl, QK_TURBO);
}

// --- turbop4_0 quantize (3-bit MSE + 1-bit QJL) ---
static __device__ void quantize_f32_turbop4_0_block(const float * __restrict__ x, block_turbop4_0 * __restrict__ y) {
    float buf[QK_TURBO];
    float residual[QK_TURBO];
    y->norm = turbo_forward_rotate_dev(x, buf, QK_TURBO);

    for (int i = 0; i < 48; i++) y->qs[i] = 0;
    for (int j = 0; j < QK_TURBO; j++) {
        int idx = turbo_nearest_3bit_dev(buf[j]);
        int bit_offset = j * 3;
        int byte_idx = bit_offset / 8;
        int bit_idx  = bit_offset % 8;
        y->qs[byte_idx] |= (uint8_t)((idx << bit_idx) & 0xFF);
        if (bit_idx > 5) {
            y->qs[byte_idx + 1] |= (uint8_t)(idx >> (8 - bit_idx));
        }
        residual[j] = buf[j] - TURBO3_CENTROIDS[idx];
    }

    // Residual L2 norm in rotated basis (needed for QJL correction in vec_dot)
    float r_norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO; j++) {
        r_norm_sq += residual[j] * residual[j];
    }
    y->r_norm = sqrtf(r_norm_sq);

    turbo_qjl_encode_dev(residual, y->qjl, QK_TURBO);
}

// --- turbop5_0 quantize (4-bit MSE + 1-bit QJL) ---
static __device__ void quantize_f32_turbop5_0_block(const float * __restrict__ x, block_turbop5_0 * __restrict__ y) {
    float buf[QK_TURBO];
    float residual[QK_TURBO];
    y->norm = turbo_forward_rotate_dev(x, buf, QK_TURBO);

    for (int i = 0; i < QK_TURBO / 2; i++) y->qs[i] = 0;
    for (int j = 0; j < QK_TURBO; j++) {
        int idx = turbo_nearest_4bit_dev(buf[j]);
        y->qs[j / 2] |= (uint8_t)(idx << (4 * (j % 2)));
        residual[j] = buf[j] - TURBO4_CENTROIDS[idx];
    }

    // Residual L2 norm in rotated basis (needed for QJL correction in vec_dot)
    float r_norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO; j++) {
        r_norm_sq += residual[j] * residual[j];
    }
    y->r_norm = sqrtf(r_norm_sq);

    turbo_qjl_encode_dev(residual, y->qjl, QK_TURBO);
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
