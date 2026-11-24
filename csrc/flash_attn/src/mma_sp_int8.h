#pragma once

#include <cstdint>

namespace FLASH_NAMESPACE {

// ─────────────────────────────────────────────────────────────────────────────
// Sparse MMA m16n8k32 : s8 × s8 → s32
//
// PTX:  mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32
//
// Per-thread register usage:
//   A (sparse, 2:4 compressed): 2 uint32 (8 int8 = 4 per half-row × 2 rows)
//   B (dense):                  2 uint32 (8 int8: first/second K-half)
//   D/C (accumulator):          4 uint32 (4 int32)
//   E (metadata):               1 uint32 (8 nibbles packed, per-thread)
//   spsel (selector):           compile-time 0 or 1
//
// Thread mapping (groupID = lane>>2, tid = lane&3):
//   tid=0: E contains 8 nibbles for row groupID      (spsel=0)
//   tid=1: E contains 8 nibbles for row groupID+8    (spsel=0)
//   tid=2: E contains 8 nibbles for row groupID      (spsel=1)
//   tid=3: E contains 8 nibbles for row groupID+8    (spsel=1)
//
// Nibble encoding: nibble = idx0 | (idx1 << 2), idx0 < idx1
//   selects which 2 of 4 elements are kept in each K-group
// ─────────────────────────────────────────────────────────────────────────────

template <int Spsel = 0>
__forceinline__ __device__ void mma_sp_m16n8k32_s32_s8_s8(
    uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
    const uint32_t a0, const uint32_t a1,
    const uint32_t b0, const uint32_t b1,
    const uint32_t meta) {
    static_assert(Spsel == 0 || Spsel == 1, "mma.sp spsel must be 0 or 1.");
    if constexpr (Spsel == 0) {
        asm volatile(
            "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, "
            "{%8, %9, %10, %11}, %12, 0x0;\n"
            : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
            : "r"(a0), "r"(a1),
              "r"(b0), "r"(b1),
              "r"(d0), "r"(d1), "r"(d2), "r"(d3),
              "r"(meta));
    } else {
        asm volatile(
            "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, "
            "{%8, %9, %10, %11}, %12, 0x1;\n"
            : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
            : "r"(a0), "r"(a1),
              "r"(b0), "r"(b1),
              "r"(d0), "r"(d1), "r"(d2), "r"(d3),
              "r"(meta));
    }
}

__forceinline__ __device__ void mma_sp_m16n8k32_s32_s8_s8_dispatch(
    uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
    const uint32_t a0, const uint32_t a1,
    const uint32_t b0, const uint32_t b1,
    const uint32_t meta, const int spsel) {
    if (spsel == 0) {
        mma_sp_m16n8k32_s32_s8_s8<0>(d0, d1, d2, d3, a0, a1, b0, b1, meta);
    } else {
        mma_sp_m16n8k32_s32_s8_s8<1>(d0, d1, d2, d3, a0, a1, b0, b1, meta);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparse MMA m16n8k32 : s8 × u8 → s32 (P_quantized × V path)
// ─────────────────────────────────────────────────────────────────────────────

template <int Spsel = 0>
__forceinline__ __device__ void mma_sp_m16n8k32_s32_s8_u8(
    uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
    const uint32_t a0, const uint32_t a1,
    const uint32_t b0, const uint32_t b1,
    const uint32_t meta) {
    static_assert(Spsel == 0 || Spsel == 1, "mma.sp spsel must be 0 or 1.");
    if constexpr (Spsel == 0) {
        asm volatile(
            "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, "
            "{%8, %9, %10, %11}, %12, 0x0;\n"
            : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
            : "r"(a0), "r"(a1),
              "r"(b0), "r"(b1),
              "r"(d0), "r"(d1), "r"(d2), "r"(d3),
              "r"(meta));
    } else {
        asm volatile(
            "mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 "
            "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, "
            "{%8, %9, %10, %11}, %12, 0x1;\n"
            : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
            : "r"(a0), "r"(a1),
              "r"(b0), "r"(b1),
              "r"(d0), "r"(d1), "r"(d2), "r"(d3),
              "r"(meta));
    }
}

__forceinline__ __device__ void mma_sp_m16n8k32_s32_s8_u8_dispatch(
    uint32_t &d0, uint32_t &d1, uint32_t &d2, uint32_t &d3,
    const uint32_t a0, const uint32_t a1,
    const uint32_t b0, const uint32_t b1,
    const uint32_t meta, const int spsel) {
    if (spsel == 0) {
        mma_sp_m16n8k32_s32_s8_u8<0>(d0, d1, d2, d3, a0, a1, b0, b1, meta);
    } else {
        mma_sp_m16n8k32_s32_s8_u8<1>(d0, d1, d2, d3, a0, a1, b0, b1, meta);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Quantization / Dequantization helpers
// ─────────────────────────────────────────────────────────────────────────────

__forceinline__ __device__ int8_t quantize_f32_to_s8(float x, float inv_scale) {
    int q = __float2int_rn(x * inv_scale);
    q = max(-128, min(127, q));
    return static_cast<int8_t>(q);
}

__forceinline__ __device__ uint8_t quantize_f32_to_u8(float x, float scale) {
    int q = __float2int_rn(x * scale);
    q = max(0, min(255, q));
    return static_cast<uint8_t>(q);
}

__forceinline__ __device__ uint32_t pack_4xf32_to_s8x4(float f0, float f1, float f2, float f3,
                                                         float inv_scale) {
    uint32_t s0 = static_cast<uint32_t>(quantize_f32_to_s8(f0, inv_scale)) & 0xFF;
    uint32_t s1 = static_cast<uint32_t>(quantize_f32_to_s8(f1, inv_scale)) & 0xFF;
    uint32_t s2 = static_cast<uint32_t>(quantize_f32_to_s8(f2, inv_scale)) & 0xFF;
    uint32_t s3 = static_cast<uint32_t>(quantize_f32_to_s8(f3, inv_scale)) & 0xFF;
    return s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
}

__forceinline__ __device__ uint32_t pack_4xf32_to_u8x4(float f0, float f1, float f2, float f3,
                                                         float scale) {
    uint32_t u0 = static_cast<uint32_t>(quantize_f32_to_u8(f0, scale));
    uint32_t u1 = static_cast<uint32_t>(quantize_f32_to_u8(f1, scale));
    uint32_t u2 = static_cast<uint32_t>(quantize_f32_to_u8(f2, scale));
    uint32_t u3 = static_cast<uint32_t>(quantize_f32_to_u8(f3, scale));
    return u0 | (u1 << 8) | (u2 << 16) | (u3 << 24);
}

__forceinline__ __device__ float dequant_s32_to_f32(int32_t val, float scale) {
    return static_cast<float>(val) * scale;
}

// Dequantize an array of int32 accumulators in-place to float.
// Used after QK^T INT8 MMA: acc_s_float = acc_s_int32 * (scale_q * scale_k)
template<int N>
__forceinline__ __device__ void dequant_acc_s32_to_f32(
    float* dst, const int32_t* src, float combined_scale) {
    #pragma unroll
    for (int i = 0; i < N; i++) {
        dst[i] = static_cast<float>(src[i]) * combined_scale;
    }
}

// Quantize float P values to int8 and pack 4 values into uint32.
// Used after softmax: P_int8 = round(P_float * scale_p_inv)
__forceinline__ __device__ uint32_t quantize_4xf32_to_s8x4(
    float v0, float v1, float v2, float v3, float inv_scale) {
    return pack_4xf32_to_s8x4(v0, v1, v2, v3, inv_scale);
}

} // namespace FLASH_NAMESPACE
