// =====================================================================
// TurboQuant quantization common definitions
// =====================================================================
//
// This header is intentionally separated from ggml-common.h so that
// changes to TurboQuant-specific structures (e.g. adding r_norm) do
// NOT cascade-rebuild every ggml-cuda file. Only files that actually
// use block_turbo*/block_turbop* should include this header.
//
// Usage mirrors ggml-common.h: the caller defines one of
//   GGML_COMMON_DECL_C / ..._CPP / ..._CUDA / ..._HIP / ...
// or the matching GGML_COMMON_IMPL_* symbol, then includes this
// header after ggml-common.h. Re-entry is allowed (no #pragma once),
// so DECL-only and IMPL-only inclusions from the same TU both work.

// -----------------------------------------------------------------
// Struct declarations (emitted when DECL is requested)
// -----------------------------------------------------------------

#ifndef GGML_COMMON_TURBO_DECL

#if defined(GGML_COMMON_DECL_C) || defined(GGML_COMMON_DECL_CPP) || \
    defined(GGML_COMMON_DECL_METAL) || defined(GGML_COMMON_DECL_CUDA) || \
    defined(GGML_COMMON_DECL_HIP) || defined(GGML_COMMON_DECL_MUSA) || \
    defined(GGML_COMMON_DECL_SYCL)

// Note: types like uint8_t and static_assert are already brought in
// by ggml-common.h which must be included before this header.

#define QK_TURBO 128  // block size = head_dim

// QJL correction constant: sqrt(pi/2)
#define TURBO_QJL_SCALE 1.2533141373155001f

// -----------------------------------------------------------------
// Algorithm 1 MSE blocks (FWHT + Lloyd-Max codebook)
// -----------------------------------------------------------------

// turbo2: 2-bit, 128 elements, 36 bytes
typedef struct {
    float   norm;                   // 4 bytes
    uint8_t qs[QK_TURBO / 4];       // 32 bytes: 2-bit indices (4 per byte)
} block_turbo2_0;
static_assert(sizeof(block_turbo2_0) == 4 + QK_TURBO / 4, "wrong turbo2_0 block size");

// turbo2h: 2.5-bit fractional, 128 elements, 40 bytes
typedef struct {
    float   norm;                   // 4 bytes
    uint8_t qs_hi[12];              // 12 bytes: 32 channels x 3-bit packed
    uint8_t qs_lo[24];              // 24 bytes: 96 channels x 2-bit packed
} block_turbo2h_0;
static_assert(sizeof(block_turbo2h_0) == 4 + 12 + 24, "wrong turbo2h_0 block size");

// turbo3: 3-bit, 128 elements, 52 bytes
typedef struct {
    float   norm;                   // 4 bytes
    uint8_t qs[48];                 // 48 bytes: 128 x 3-bit packed
} block_turbo3_0;
static_assert(sizeof(block_turbo3_0) == 4 + 48, "wrong turbo3_0 block size");

// turbo3h: 3.5-bit fractional, 128 elements, 60 bytes
typedef struct {
    float   norm;                   // 4 bytes
    uint8_t qs_hi[32];              // 32 bytes: 64 channels x 4-bit packed
    uint8_t qs_lo[24];              // 24 bytes: 64 channels x 3-bit packed
} block_turbo3h_0;
static_assert(sizeof(block_turbo3h_0) == 4 + 32 + 24, "wrong turbo3h_0 block size");

// turbo4: 4-bit, 128 elements, 68 bytes
typedef struct {
    float   norm;                   // 4 bytes
    uint8_t qs[QK_TURBO / 2];       // 64 bytes: 4-bit indices
} block_turbo4_0;
static_assert(sizeof(block_turbo4_0) == 4 + QK_TURBO / 2, "wrong turbo4_0 block size");

// -----------------------------------------------------------------
// Algorithm 2 prod blocks (MSE + QJL residual)
// -----------------------------------------------------------------
//
// r_norm is the L2 norm of the residual in the rotated unit-sphere basis,
// needed by the vec_dot QJL correction formula in fattn-common.cuh.

// turbop3: 3-bit prod = 2-bit MSE + 1-bit QJL, 56 bytes
typedef struct {
    float   norm;                   // 4 bytes
    float   r_norm;                 // 4 bytes: residual L2 norm in rotated basis
    uint8_t qs[QK_TURBO / 4];       // 32 bytes: 2-bit MSE indices
    uint8_t qjl[QK_TURBO / 8];      // 16 bytes: 1-bit QJL signs
} block_turbop3_0;
static_assert(sizeof(block_turbop3_0) == 8 + QK_TURBO / 4 + QK_TURBO / 8, "wrong turbop3_0 block size");

// turbop4: 4-bit prod = 3-bit MSE + 1-bit QJL, 72 bytes
typedef struct {
    float   norm;                   // 4 bytes
    float   r_norm;                 // 4 bytes
    uint8_t qs[48];                 // 48 bytes: 3-bit MSE indices
    uint8_t qjl[QK_TURBO / 8];      // 16 bytes: 1-bit QJL signs
} block_turbop4_0;
static_assert(sizeof(block_turbop4_0) == 8 + 48 + QK_TURBO / 8, "wrong turbop4_0 block size");

// turbop5: 5-bit prod = 4-bit MSE + 1-bit QJL, 88 bytes
typedef struct {
    float   norm;                   // 4 bytes
    float   r_norm;                 // 4 bytes
    uint8_t qs[QK_TURBO / 2];       // 64 bytes: 4-bit MSE indices
    uint8_t qjl[QK_TURBO / 8];      // 16 bytes: 1-bit QJL signs
} block_turbop5_0;
static_assert(sizeof(block_turbop5_0) == 8 + QK_TURBO / 2 + QK_TURBO / 8, "wrong turbop5_0 block size");

#define GGML_COMMON_TURBO_DECL  // set guard ONLY after structs are emitted
#endif // defined(GGML_COMMON_DECL_*)

#endif // GGML_COMMON_TURBO_DECL

////////////////////////////////////////////////////////////////////////////////

// -----------------------------------------------------------------
// Lloyd-Max codebook tables (emitted when IMPL is requested)
// -----------------------------------------------------------------

#ifndef GGML_COMMON_TURBO_IMPL

// Only emit the tables if the caller has set GGML_COMMON_IMPL (which
// ggml-common.h sets as a side effect of one of GGML_COMMON_IMPL_*).
#if defined(GGML_COMMON_IMPL)

GGML_TABLE_BEGIN(float, TURBO2_CENTROIDS, 4)
    -1.5104176085f, -0.4527800346f, +0.4527800346f, +1.5104176085f
GGML_TABLE_END()

GGML_TABLE_BEGIN(float, TURBO3_CENTROIDS, 8)
    -2.1519457045f, -1.3439092785f, -0.7560052812f, -0.2450941789f,
    +0.2450941789f, +0.7560052812f, +1.3439092785f, +2.1519457045f
GGML_TABLE_END()

GGML_TABLE_BEGIN(float, TURBO4_CENTROIDS, 16)
    -2.7325895588f, -2.0690172128f, -1.6180463720f, -1.2562311842f,
    -0.9423404451f, -0.6567591097f, -0.3880482939f, -0.1283950280f,
    +0.1283950280f, +0.3880482939f, +0.6567591097f, +0.9423404451f,
    +1.2562311842f, +1.6180463720f, +2.0690172128f, +2.7325895588f
GGML_TABLE_END()

GGML_TABLE_BEGIN(float, TURBO2_BOUNDARIES, 3)
    -0.9815988216f, 0.0f, +0.9815988216f
GGML_TABLE_END()

GGML_TABLE_BEGIN(float, TURBO3_BOUNDARIES, 7)
    -1.7479274915f, -1.0499572799f, -0.5005497301f, 0.0f,
    +0.5005497301f, +1.0499572799f, +1.7479274915f
GGML_TABLE_END()

GGML_TABLE_BEGIN(float, TURBO4_BOUNDARIES, 15)
    -2.4008033858f, -1.8435317924f, -1.4371387781f, -1.0992858146f,
    -0.7995497774f, -0.5224037018f, -0.2582216609f, 0.0f,
    +0.2582216609f, +0.5224037018f, +0.7995497774f, +1.0992858146f,
    +1.4371387781f, +1.8435317924f, +2.4008033858f
GGML_TABLE_END()

#define GGML_COMMON_TURBO_IMPL  // set guard ONLY after tables are emitted
#endif // defined(GGML_COMMON_IMPL)

#endif // GGML_COMMON_TURBO_IMPL
