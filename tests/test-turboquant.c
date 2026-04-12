// Unit tests for TurboQuant quantize/dequantize round-trip.
//
// Verifies MSE-per-coordinate and cosine similarity for all 8 TurboQuant
// GGML types (Algorithm 1 MSE + Algorithm 2 prod/QJL) against the bounds
// reported in the TurboQuant paper (arXiv:2504.19874).
//
// This test uses the internal reference CPU quant functions from
// ggml/src/ggml-quants.h and block structs from ggml-common.h, so the
// target must add ggml/src to its include path.

#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-common.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N_BLOCKS   200
#define BLOCK_SIZE QK_TURBO  // 128

// ----------------------------------------------------------------------------
// Test helpers
// ----------------------------------------------------------------------------

// Simple xorshift32 so the test is deterministic and independent of rand().
static uint32_t xs_state = 0xC0FFEEu;
static uint32_t xs_next(void) {
    uint32_t x = xs_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    xs_state = x;
    return x;
}
static float xs_uniform(void) {
    // uniform in (0,1]
    return ((xs_next() >> 8) + 1) * (1.0f / 16777216.0f);
}
static float xs_normal(void) {
    // Box-Muller
    float u1 = xs_uniform();
    float u2 = xs_uniform();
    float r  = sqrtf(-2.0f * logf(u1));
    float th = 6.28318530717958647692f * u2;
    return r * cosf(th);
}

// Generate a random vector with N(0,1) entries scaled to have a random norm
// drawn log-uniformly in [0.25, 4.0]. This exercises both unit-ish and
// larger-magnitude inputs, matching what KV-cache rows look like in practice.
static void gen_random_vec(float * v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = xs_normal();
    }
    // normalize and rescale
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)v[i] * v[i];
    float inv = (s > 0.0) ? (float)(1.0 / sqrt(s)) : 1.0f;

    // log-uniform norm in [0.25, 4.0]
    float u = xs_uniform();
    float target_norm = powf(2.0f, -2.0f + 4.0f * u);

    float k = inv * target_norm;
    for (int i = 0; i < n; i++) v[i] *= k;
}

static double compute_sse(const float * a, const float * b, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        s += d * d;
    }
    return s;
}

static double compute_cosine(const float * a, const float * b, int n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

// ----------------------------------------------------------------------------
// Generic round-trip harness
// ----------------------------------------------------------------------------

typedef void (*quant_fn)  (const float * x, void * y, int64_t k);
typedef void (*dequant_fn)(const void  * x, float * y, int64_t k);

typedef struct {
    const char * name;
    quant_fn     quant;
    dequant_fn   dequant;
    size_t       block_bytes;
    // Expected bounds from the TurboQuant paper (d=128 Beta-optimal Lloyd-Max).
    // Note: MSE/coord is computed on UNIT-NORM vectors so tol is absolute on
    // that scale; we rescale measured MSE by 1/||x||^2 below to compare.
    float        mse_per_coord_bound;
    float        cosine_bound;
} turbo_type_t;

// Thin wrappers so we can stuff the typed ref fns into void* signatures
// without tripping the C strict-prototype / aliasing warnings.
#define WRAP_PAIR(NAME, BLOCK_T)                                                   \
    static void q_##NAME(const float * x, void * y, int64_t k) {                   \
        quantize_row_##NAME##_ref(x, (BLOCK_T *)y, k);                             \
    }                                                                              \
    static void d_##NAME(const void * x, float * y, int64_t k) {                   \
        dequantize_row_##NAME((const BLOCK_T *)x, y, k);                           \
    }

WRAP_PAIR(turbo2_0,  block_turbo2_0)
WRAP_PAIR(turbo2h_0, block_turbo2h_0)
WRAP_PAIR(turbo3_0,  block_turbo3_0)
WRAP_PAIR(turbo3h_0, block_turbo3h_0)
WRAP_PAIR(turbo4_0,  block_turbo4_0)
WRAP_PAIR(turbop3_0, block_turbop3_0)
WRAP_PAIR(turbop4_0, block_turbop4_0)
WRAP_PAIR(turbop5_0, block_turbop5_0)

static int test_turbo_type(const turbo_type_t * t) {
    float  x[BLOCK_SIZE];
    float  y[BLOCK_SIZE];
    uint8_t blk[256]; // large enough for any turbo block (<= 100 bytes)
    if (t->block_bytes > sizeof(blk)) {
        fprintf(stderr, "FATAL: block buffer too small for %s (%zu bytes)\n",
                t->name, t->block_bytes);
        return 1;
    }

    double sum_mse_per_coord = 0.0; // normalized MSE (mse / ||x||^2)
    double sum_cosine        = 0.0;
    double worst_cosine      = 1.0;

    for (int b = 0; b < N_BLOCKS; b++) {
        gen_random_vec(x, BLOCK_SIZE);

        memset(blk, 0, t->block_bytes);
        t->quant  (x,   (void *)blk, BLOCK_SIZE);
        t->dequant((const void *)blk, y, BLOCK_SIZE);

        double sse    = compute_sse(x, y, BLOCK_SIZE);
        double norm_sq = 0.0;
        for (int i = 0; i < BLOCK_SIZE; i++) norm_sq += (double)x[i] * x[i];
        if (norm_sq <= 0.0) norm_sq = 1.0;

        // TurboQuant paper reports MSE per coordinate for unit-norm vectors.
        // For a scaled vector v = alpha * u with ||u||=1, per-coord MSE scales
        // as alpha^2, so we divide sse by ||v||^2 * (1/d) is equivalent to
        // (sse / d) / alpha^2 -> (sse / (d * alpha^2)) = (sse / (d * norm_sq / d))
        // -> sse / norm_sq. So normalized "MSE per coord on unit scale"
        // = (sse / norm_sq).
        double mse_per_coord_unit = sse / norm_sq;

        double cos = compute_cosine(x, y, BLOCK_SIZE);

        sum_mse_per_coord += mse_per_coord_unit;
        sum_cosine        += cos;
        if (cos < worst_cosine) worst_cosine = cos;
    }

    double avg_mse = sum_mse_per_coord / (double)N_BLOCKS;
    double avg_cos = sum_cosine        / (double)N_BLOCKS;

    int ok_mse = (avg_mse <= t->mse_per_coord_bound);
    int ok_cos = (avg_cos >= t->cosine_bound);
    int ok     = ok_mse && ok_cos;

    printf("  %-9s  block=%3zuB  avg_mse/coord=%.5f (bound %.5f)  "
           "avg_cos=%.5f  worst_cos=%.5f (bound %.3f)   %s\n",
           t->name, t->block_bytes,
           avg_mse, t->mse_per_coord_bound,
           avg_cos, worst_cosine, t->cosine_bound,
           ok ? "PASS" : "FAIL");

    if (!ok) {
        if (!ok_mse) {
            printf("    -> MSE/coord %.5f exceeds bound %.5f\n",
                   avg_mse, t->mse_per_coord_bound);
        }
        if (!ok_cos) {
            printf("    -> avg cosine %.5f below bound %.3f\n",
                   avg_cos, t->cosine_bound);
        }
    }
    return ok ? 0 : 1;
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main(void) {
    printf("\n=== TurboQuant Round-Trip Tests ===\n");
    printf("blocks/type = %d, block size = %d, seed = xorshift32(0xC0FFEE)\n",
           N_BLOCKS, BLOCK_SIZE);

    // Tolerance notes:
    //   - Paper reference MSE/coord for Beta-d=128 Lloyd-Max:
    //       2-bit ~ 0.117,  3-bit ~ 0.034,  4-bit ~ 0.009
    //   - Fractional bits (2.5, 3.5) interpolate, with a small overhead.
    //   - prod types add 1-bit QJL on residual: per-coord MSE should be no
    //     worse than the underlying (b-1)-bit MSE stage, but we allow slack
    //     because the residual is noisy and Lloyd-Max bounds only apply
    //     tightly to FWHT-rotated Beta inputs.
    //   - We use ~1.25x the paper value as a conservative pass bound to
    //     absorb finite-sample noise (N_BLOCKS=200) and Box-Muller tail.
    //
    //   Cosine bounds follow the CLAUDE.md spec (0.94 / 0.98 / 0.995) with
    //   small relaxations for fractional and prod variants.
    const turbo_type_t types[] = {
        // MSE (Algorithm 1)
        { "turbo2",  q_turbo2_0,  d_turbo2_0,  sizeof(block_turbo2_0),  0.150f, 0.940f },
        { "turbo2h", q_turbo2h_0, d_turbo2h_0, sizeof(block_turbo2h_0), 0.100f, 0.950f },
        { "turbo3",  q_turbo3_0,  d_turbo3_0,  sizeof(block_turbo3_0),  0.045f, 0.980f },
        { "turbo3h", q_turbo3h_0, d_turbo3h_0, sizeof(block_turbo3h_0), 0.025f, 0.988f },
        { "turbo4",  q_turbo4_0,  d_turbo4_0,  sizeof(block_turbo4_0),  0.012f, 0.995f },
        // prod (Algorithm 2): tol is relaxed because QJL is unbiased for
        // inner products but high-variance on reconstruction.
        { "turbop3", q_turbop3_0, d_turbop3_0, sizeof(block_turbop3_0), 0.150f, 0.940f },
        { "turbop4", q_turbop4_0, d_turbop4_0, sizeof(block_turbop4_0), 0.050f, 0.975f },
        { "turbop5", q_turbop5_0, d_turbop5_0, sizeof(block_turbop5_0), 0.020f, 0.990f },
    };
    const int n_types = (int)(sizeof(types) / sizeof(types[0]));

    int failed = 0;
    for (int i = 0; i < n_types; i++) {
        // Reset RNG per type so every type sees the same input sequence.
        xs_state = 0xC0FFEEu;
        failed += test_turbo_type(&types[i]);
    }

    printf("\n%s: %d / %d types passed\n",
           failed == 0 ? "OK" : "FAIL",
           n_types - failed, n_types);

    return failed == 0 ? 0 : 1;
}
