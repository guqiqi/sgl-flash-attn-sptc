/******************************************************************************
 * INT8 sptc SPTC sparse attention kernel for blockM=32.
 *
 * Full INT8 pipeline with sparse pair path:
 *   QK^T: CUTE TiledMma SM80_16x8x32_S32S8S8S32_TN -> int32 -> dequant -> float
 *   Softmax: online softmax in float
 *   Sparse pair: extract_sparse_pair -> P quantize(int8)
 *     -> mma.sp.sync.m16n8k32 (per-thread metadata) -> int32 -> dequant -> float
 *   Dense block/column path: same as v1 INT8
 *
 * Pair geometry: blockM=32, pairN=64, halfN=32.
 * Metadata from sparse_metadata_int8_blockm32.h (per-thread, non-ordered).
 ******************************************************************************/

#pragma once

#include "flash_fwd_kernel.h"
#include "kernel_traits_int8.h"
#include "mma_sp_int8.h"
#include "sparse_metadata_int8_blockm32.h"
#include "sparse_choice_metadata_bm0_31_blockm32.h"

namespace FLASH_NAMESPACE {

using namespace cute;

namespace flash_sptc_int8_blockm32_impl {

constexpr int kSparseBlockM = 32;
constexpr int kSparsePairBlockN = 64;
constexpr int kSparseHalfBlockN = 32;

// ─────────────────────────────────────────────────────────────────────
// INT8 SPTC: V row permutation for mma.sp m16n8k32
// ─────────────────────────────────────────────────────────────────────
// Combined step1 (odd31 swap) + INT8 step2 (32-col TIG grouping).
// Maps permuted pair-row index → original pair-row index.
// Bit permutation (involution): swap bits {0↔5, 1↔3, 2↔4}.
// o[5:0] = {i[0], i[4], i[1], i[3], i[2], i[5]}
__device__ __forceinline__ int combined_v_row_map_int8(int i) {
    return ((i & 1) << 5) | (((i >> 3) & 1) << 1) | (((i >> 4) & 1) << 2)
         | (((i >> 1) & 1) << 3) | (((i >> 2) & 1) << 4) | (((i >> 5) & 1) << 0);
}

__device__ constexpr int kVRowInvPermute_Int8_BlockM32[64] = {
     0, 32,  8, 40, 16, 48, 24, 56,  2, 34, 10, 42, 18, 50, 26, 58,
     4, 36, 12, 44, 20, 52, 28, 60,  6, 38, 14, 46, 22, 54, 30, 62,
     1, 33,  9, 41, 17, 49, 25, 57,  3, 35, 11, 43, 19, 51, 27, 59,
     5, 37, 13, 45, 21, 53, 29, 61,  7, 39, 15, 47, 23, 55, 31, 63,
};

// ─────────────────────────────────────────────────────────────────────
// INT8 SPTC: extract_sparse_pair for mma.sp m16n8k32
// ─────────────────────────────────────────────────────────────────────
// The INT8 step2 permutation groups 32 pair-cols into 8 K-groups of 4,
// where consecutive pairs of groups belong to the same TIG:
//   TIG0: G0{0,1,8,9}, G1{16,17,24,25}
//   TIG1: G2{2,3,10,11}, G3{18,19,26,27}
//   TIG2: G4{4,5,12,13}, G5{20,21,28,29}
//   TIG3: G6{6,7,14,15}, G7{22,23,30,31}
//
// For mma.sp m16n8k32: each TIG provides a0,a1 = 2×uint32 = 8 int8.
// a0 = row0 compressed cols 0-3 (2 from G_even, 2 from G_odd)
// a1 = row1 compressed cols 0-3.
//
// The FP16 choice metadata table (from sparse_choice_metadata_bm0_31_blockm32.h)
// is STILL used for the extract step because it operates on float acc_s
// fragments indexed by the FP16-compatible (t,k) pattern.
// The actual INT8 mma.sp metadata (per-thread nibbles) was regenerated
// with INT8-specific step2 in gen_int8_metadata.py.

template <int DstT, int DstK, typename TensorSrc, typename TensorDst>
__forceinline__ __device__ void extract_sparse_pair_element_int8(
    const TensorSrc &src_fragment, TensorDst &dst_fragment,
    const uint32_t metadata_word, const int lane_group_col,
    const bool zero_left_half) {
    // INT8 step2 mapping within a 32-col k-half:
    // DstT: 0..3 (4 values in one mma.sp A tile: 2 rows × 2 compressed cols)
    // DstK: 0..1 (2 k-halves per pair: k_iter=0 cols 0-31, k_iter=1 cols 32-63)
    //
    // Within each k-half (32 cols = 4 N-tiles of 8 cols):
    // TIG's data comes from 2 groups of 4 consecutive cols.
    // Group A: N-tiles 0,1 (src_k = 0..1 in DstK's N-tile range)
    // Group B: N-tiles 2,3 (src_k = 2..3)
    //
    // DstT mapping to (row, compressed_col):
    // DstT=0: row0, comp_col 0 (from group A)
    // DstT=1: row0, comp_col 1 (from group A)
    // DstT=2: row1, comp_col 0 (from group A)
    // DstT=3: row1, comp_col 1 (from group A)
    // (group B selections go into comp_cols 2,3)
    //
    // But with 8 nibbles in the metadata, each nibble selects 2 of 4 from one group.
    // For k_iter DstK: nibbles 0-7 cover groups 0-7 of the permuted 32-col half.
    // TIG0 cares about nibbles for G0 and G1 (indices 0,1).
    //
    // However, ALL threads hold ALL 8 nibbles (metadata is shared).
    // Each TIG reads its own group's nibble and uses it to select from its fragment.
    
    // Simplified: use the INT8 mma.sp per-thread metadata directly.
    // Each thread already has the correct metadata for its assigned row/K-half.
    // The extract just needs to:
    // 1. Read 4 float values from the C fragment for group A (N-tiles 0,1)
    // 2. Read 4 float values from the C fragment for group B (N-tiles 2,3)
    // 3. Apply nibble-based selection (2 from each group = 4 selected per row)
    // 4. Quantize to int8 and pack
    
    // For the cute C fragment: src_fragment is indexed as (t, 0, k) where
    // t = val_within_tile (0..3), k = n_tile (0..7 for pair 64 cols)
    // PTX: d0=(row0,col_even), d1=(row0,col_odd), d2=(row1,col_even), d3=(row1,col_odd)
    // BUT cute val ordering differs from PTX register ordering!
    // We must use the correct cute index to access the right register.
    
    // For this function, we match the FP16 pattern but adjust for INT8 step2.
    // src_t: which val index in the C fragment tile
    // src_k: which N-tile
    constexpr int src_t       = (DstT & 2) | (DstK >> 1);
    constexpr int src_k_left  = (DstK & 1) * 2 + (DstT & 1);
    constexpr int src_k_right = src_k_left + 4;
    constexpr int J           = DstT >> 1;
    constexpr int bit_offset_base = J * 16 + (DstT & 1) * 2;

    const int select = (metadata_word >> (bit_offset_base + lane_group_col * 4)) & 1;
    if (select == 0) {
        if (zero_left_half) {
            dst_fragment(Int<DstT>{}, Int<0>{}, Int<DstK>{}) = -INFINITY;
        } else {
            dst_fragment(Int<DstT>{}, Int<0>{}, Int<DstK>{}) =
                src_fragment(Int<src_t>{}, Int<0>{}, Int<src_k_left>{});
        }
    } else {
        dst_fragment(Int<DstT>{}, Int<0>{}, Int<DstK>{}) =
            src_fragment(Int<src_t>{}, Int<0>{}, Int<src_k_right>{});
    }
}

template <typename TensorSrc, typename TensorDst>
__forceinline__ __device__ void extract_sparse_pair_int8(
    const TensorSrc &src_fragment, TensorDst &dst_fragment,
    const uint32_t *metadata_row, const bool zero_left_half = false) {
    const int lane_group_col = threadIdx.x & 3;
#define EXTRACT_ELEMENT_INT8(DT, DK) \
    extract_sparse_pair_element_int8<DT, DK>(src_fragment, dst_fragment, metadata_row[DK], lane_group_col, zero_left_half)
    EXTRACT_ELEMENT_INT8(0, 0); EXTRACT_ELEMENT_INT8(1, 0); EXTRACT_ELEMENT_INT8(2, 0); EXTRACT_ELEMENT_INT8(3, 0);
    EXTRACT_ELEMENT_INT8(0, 1); EXTRACT_ELEMENT_INT8(1, 1); EXTRACT_ELEMENT_INT8(2, 1); EXTRACT_ELEMENT_INT8(3, 1);
    EXTRACT_ELEMENT_INT8(0, 2); EXTRACT_ELEMENT_INT8(1, 2); EXTRACT_ELEMENT_INT8(2, 2); EXTRACT_ELEMENT_INT8(3, 2);
    EXTRACT_ELEMENT_INT8(0, 3); EXTRACT_ELEMENT_INT8(1, 3); EXTRACT_ELEMENT_INT8(2, 3); EXTRACT_ELEMENT_INT8(3, 3);
#undef EXTRACT_ELEMENT_INT8
}

// ─────────────────────────────────────────────────────────────────────
// INT8 SPTC: Sparse PV GEMM using mma.sp m16n8k32
// ─────────────────────────────────────────────────────────────────────
//
// V load strategy is selected at compile time:
//
//   FLASH_INT8_VLOAD_BYTE (default / mode A):
//     Per-byte load from smem, correct on all architectures.
//     Each uint32 B-operand = 4 int8 from same N-col, 4 consecutive K-rows.
//     Slower due to 8 individual byte loads per n_tile.
//
//   FLASH_INT8_VLOAD_LDMATRIX_B16 (mode B):
//     Use ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 (same as FP16).
//     Treats V smem as if it were half_t and loads 2-byte granules.
//     INCORRECT for INT8 MMA (b16 transpose ≠ b8 transpose), but useful
//     to measure the performance ceiling if ldmatrix were available.
//     Requires V smem layout compatible with FP16 ldmatrix (Swizzle<3,3,4>).
//
//   FLASH_INT8_VLOAD_LDMATRIX_B8 (mode C):
//     Use ldmatrix.sync.aligned.x4.trans.m8n8.shared.b8 (sm_120f+ only).
//     Correct INT8 B-operand layout. Will not compile on sm_80.
//     Prepared for future Blackwell+ targets.
//
// To select: pass -DFLASH_INT8_VLOAD_LDMATRIX_B16 or
//            -DFLASH_INT8_VLOAD_LDMATRIX_B8 at compile time.
//            Default (neither defined) = per-byte load.

template<int kSwizzle, int kSwizzleS = 3,
         typename Tensor0, typename Tensor1>
__forceinline__ __device__ void gemm_rs_sparse_mma_sp_int8(
    Tensor0 &acc_o, Tensor1 &rA, const int8_t *v_smem_base,
    const uint32_t meta_word, float pv_scale, int v_stride) {

    constexpr int kKtiles = decltype(size<2>(rA))::value;
    constexpr int kNtiles = decltype(size<2>(acc_o))::value;

    const int lane_id = threadIdx.x & 31;

    #pragma unroll
    for (int k_iter = 0; k_iter < kKtiles; ++k_iter) {
        auto a_frag = rA(_, _, k_iter);
        auto a_regs = cute::recast<uint32_t>(a_frag);
        const uint32_t a0 = a_regs(0);
        const uint32_t a1 = a_regs(1);
        const int spsel = k_iter & 1;

#if defined(FLASH_INT8_VLOAD_LDMATRIX_B16)
        // ── Mode B: ldmatrix.b16.trans (INCORRECT for INT8 but fast) ──
        // Uses ldmatrix with b16 transpose to measure V-load perf ceiling.
        // Data is incorrect but load + MMA throughput is representative.
        //
        // INT8 V smem: row_stride = v_stride (128 bytes for headdim=128).
        // ldmatrix.x4.trans.b16 needs 16-byte-aligned address per thread.
        // We skip swizzle and use raw row_base + col_start (multiples of 16).
        // Row mapping still uses combined_v_row_map_int8 for realistic ALU.
        {
            const int lane_row = lane_id & 15;
            const int row_mapped = combined_v_row_map_int8(k_iter * 32 + lane_row);
            const int row_base = row_mapped * v_stride;

            #pragma unroll
            for (int n_pair = 0; n_pair < kNtiles / 2; ++n_pair) {
                // col_start in bytes, 16B-aligned for ldmatrix
                const int col_start = n_pair * 16;
                const int offset = row_base + col_start;
                const uint32_t smem_addr = cast_smem_ptr_to_uint(v_smem_base + offset);

                uint32_t b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(smem_addr));

                // N-tile 0
                {
                    uint32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    mma_sp_m16n8k32_s32_s8_s8_dispatch(
                        d0, d1, d2, d3, a0, a1, b0, b1, meta_word, spsel);
#ifdef FLASH_INT8_SKIP_DEQUANT
                    acc_o(0, _0{}, n_pair * 2 + 0) += reinterpret_cast<float&>(d0);
                    acc_o(1, _0{}, n_pair * 2 + 0) += reinterpret_cast<float&>(d1);
                    acc_o(2, _0{}, n_pair * 2 + 0) += reinterpret_cast<float&>(d2);
                    acc_o(3, _0{}, n_pair * 2 + 0) += reinterpret_cast<float&>(d3);
#else
                    acc_o(0, _0{}, n_pair * 2 + 0) += static_cast<float>(reinterpret_cast<int32_t&>(d0)) * pv_scale;
                    acc_o(1, _0{}, n_pair * 2 + 0) += static_cast<float>(reinterpret_cast<int32_t&>(d1)) * pv_scale;
                    acc_o(2, _0{}, n_pair * 2 + 0) += static_cast<float>(reinterpret_cast<int32_t&>(d2)) * pv_scale;
                    acc_o(3, _0{}, n_pair * 2 + 0) += static_cast<float>(reinterpret_cast<int32_t&>(d3)) * pv_scale;
#endif
                }
                // N-tile 1
                {
                    uint32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    mma_sp_m16n8k32_s32_s8_s8_dispatch(
                        d0, d1, d2, d3, a0, a1, b2, b3, meta_word, spsel);
#ifdef FLASH_INT8_SKIP_DEQUANT
                    acc_o(0, _0{}, n_pair * 2 + 1) += reinterpret_cast<float&>(d0);
                    acc_o(1, _0{}, n_pair * 2 + 1) += reinterpret_cast<float&>(d1);
                    acc_o(2, _0{}, n_pair * 2 + 1) += reinterpret_cast<float&>(d2);
                    acc_o(3, _0{}, n_pair * 2 + 1) += reinterpret_cast<float&>(d3);
#else
                    acc_o(0, _0{}, n_pair * 2 + 1) += static_cast<float>(reinterpret_cast<int32_t&>(d0)) * pv_scale;
                    acc_o(1, _0{}, n_pair * 2 + 1) += static_cast<float>(reinterpret_cast<int32_t&>(d1)) * pv_scale;
                    acc_o(2, _0{}, n_pair * 2 + 1) += static_cast<float>(reinterpret_cast<int32_t&>(d2)) * pv_scale;
                    acc_o(3, _0{}, n_pair * 2 + 1) += static_cast<float>(reinterpret_cast<int32_t&>(d3)) * pv_scale;
#endif
                }
            }
        }

#elif defined(FLASH_INT8_VLOAD_LDMATRIX_B8)
        // ── Mode C: ldmatrix.b8.trans (sm_120f+ only, CORRECT for INT8) ──
        // ldmatrix.sync.aligned.x4.trans.m8n8.shared.b8 transposes at byte
        // granularity: each uint32 output = 4 bytes from same column, 4 rows.
        // This is exactly what mma.sp.m16n8k32 B-operand expects.
        {
            const int lane_row = lane_id & 15;
            const int lane_col_half = lane_id >> 4;
            const int row_mapped = combined_v_row_map_int8(k_iter * 32 + lane_row);
            const int row_base = row_mapped * v_stride;
            // Swizzle for b8: using same convention as byte layout
            constexpr int kSwMask = ((1 << kSwizzle) - 1) << kSwizzleS;
            const int y_bits = (row_mapped >> kSwizzleS) & ((1 << kSwizzle) - 1);

            #pragma unroll
            for (int n_pair = 0; n_pair < kNtiles / 2; ++n_pair) {
                const int col_idx = n_pair * 16 + lane_col_half * 8;
                const int col_lo_bits = col_idx & ~kSwMask;
                const int z_bits = (col_idx >> kSwizzleS) & ((1 << kSwizzle) - 1);
                const int swizzled_col = col_lo_bits | ((z_bits ^ y_bits) << kSwizzleS);
                const int offset = row_base + swizzled_col;
                const uint32_t smem_addr = cast_smem_ptr_to_uint(v_smem_base + offset);

                uint32_t b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b8 {%0, %1, %2, %3}, [%4];\n"
                    : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(smem_addr));

                // N-tile 0
                {
                    uint32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    mma_sp_m16n8k32_s32_s8_s8_dispatch(
                        d0, d1, d2, d3, a0, a1, b0, b1, meta_word, spsel);
#ifdef FLASH_INT8_SKIP_DEQUANT
                    acc_o(0, _0{}, n_pair * 2 + 0) += reinterpret_cast<float&>(d0);
                    acc_o(1, _0{}, n_pair * 2 + 0) += reinterpret_cast<float&>(d1);
                    acc_o(2, _0{}, n_pair * 2 + 0) += reinterpret_cast<float&>(d2);
                    acc_o(3, _0{}, n_pair * 2 + 0) += reinterpret_cast<float&>(d3);
#else
                    acc_o(0, _0{}, n_pair * 2 + 0) += static_cast<float>(reinterpret_cast<int32_t&>(d0)) * pv_scale;
                    acc_o(1, _0{}, n_pair * 2 + 0) += static_cast<float>(reinterpret_cast<int32_t&>(d1)) * pv_scale;
                    acc_o(2, _0{}, n_pair * 2 + 0) += static_cast<float>(reinterpret_cast<int32_t&>(d2)) * pv_scale;
                    acc_o(3, _0{}, n_pair * 2 + 0) += static_cast<float>(reinterpret_cast<int32_t&>(d3)) * pv_scale;
#endif
                }
                // N-tile 1
                {
                    uint32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    mma_sp_m16n8k32_s32_s8_s8_dispatch(
                        d0, d1, d2, d3, a0, a1, b2, b3, meta_word, spsel);
#ifdef FLASH_INT8_SKIP_DEQUANT
                    acc_o(0, _0{}, n_pair * 2 + 1) += reinterpret_cast<float&>(d0);
                    acc_o(1, _0{}, n_pair * 2 + 1) += reinterpret_cast<float&>(d1);
                    acc_o(2, _0{}, n_pair * 2 + 1) += reinterpret_cast<float&>(d2);
                    acc_o(3, _0{}, n_pair * 2 + 1) += reinterpret_cast<float&>(d3);
#else
                    acc_o(0, _0{}, n_pair * 2 + 1) += static_cast<float>(reinterpret_cast<int32_t&>(d0)) * pv_scale;
                    acc_o(1, _0{}, n_pair * 2 + 1) += static_cast<float>(reinterpret_cast<int32_t&>(d1)) * pv_scale;
                    acc_o(2, _0{}, n_pair * 2 + 1) += static_cast<float>(reinterpret_cast<int32_t&>(d2)) * pv_scale;
                    acc_o(3, _0{}, n_pair * 2 + 1) += static_cast<float>(reinterpret_cast<int32_t&>(d3)) * pv_scale;
#endif
                }
            }
        }

#else
        // ── Mode A (default): per-byte load from smem (correct, all archs) ──
        {
            constexpr int kSwMask = ((1 << kSwizzle) - 1) << kSwizzleS;
            constexpr int kSwMaskInv = ~kSwMask;
            const int groupID = lane_id >> 2;
            const int tid_in_group = lane_id & 3;

            int row_base_lo[4], row_base_hi[4];
            int y_lo[4], y_hi[4];
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int k_lo = k_iter * 32 + tid_in_group * 4 + i;
                const int vr_lo = combined_v_row_map_int8(k_lo);
                const int vr_hi = combined_v_row_map_int8(k_lo + 16);
                row_base_lo[i] = vr_lo * v_stride;
                row_base_hi[i] = vr_hi * v_stride;
                y_lo[i] = (vr_lo >> kSwizzleS) & ((1 << kSwizzle) - 1);
                y_hi[i] = (vr_hi >> kSwizzleS) & ((1 << kSwizzle) - 1);
            }

            #pragma unroll
            for (int n_tile = 0; n_tile < kNtiles; ++n_tile) {
                const int col = n_tile * 8 + groupID;
                const int col_lo_bits = col & kSwMaskInv;
                const int z_bits = (col >> kSwizzleS) & ((1 << kSwizzle) - 1);

                int8_t b0_bytes[4], b1_bytes[4];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int sw_lo = col_lo_bits | ((z_bits ^ y_lo[i]) << kSwizzleS);
                    const int sw_hi = col_lo_bits | ((z_bits ^ y_hi[i]) << kSwizzleS);
                    b0_bytes[i] = v_smem_base[row_base_lo[i] + sw_lo];
                    b1_bytes[i] = v_smem_base[row_base_hi[i] + sw_hi];
                }
                uint32_t b0, b1;
                __builtin_memcpy(&b0, b0_bytes, 4);
                __builtin_memcpy(&b1, b1_bytes, 4);

                uint32_t d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                mma_sp_m16n8k32_s32_s8_s8_dispatch(
                    d0, d1, d2, d3, a0, a1, b0, b1, meta_word, spsel);
#ifdef FLASH_INT8_SKIP_DEQUANT
                acc_o(0, _0{}, n_tile) += reinterpret_cast<float&>(d0);
                acc_o(1, _0{}, n_tile) += reinterpret_cast<float&>(d1);
                acc_o(2, _0{}, n_tile) += reinterpret_cast<float&>(d2);
                acc_o(3, _0{}, n_tile) += reinterpret_cast<float&>(d3);
#else
                acc_o(0, _0{}, n_tile) += static_cast<float>(reinterpret_cast<int32_t&>(d0)) * pv_scale;
                acc_o(1, _0{}, n_tile) += static_cast<float>(reinterpret_cast<int32_t&>(d1)) * pv_scale;
                acc_o(2, _0{}, n_tile) += static_cast<float>(reinterpret_cast<int32_t&>(d2)) * pv_scale;
                acc_o(3, _0{}, n_tile) += static_cast<float>(reinterpret_cast<int32_t&>(d3)) * pv_scale;
#endif
            }
        }
#endif // FLASH_INT8_VLOAD_*
    }
}

} // namespace flash_sptc_int8_blockm32_impl

////////////////////////////////////////////////////////////////////////////////////////////////////
// Main sptc INT8 blockM=32 kernel
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi,
         bool Is_even_MN, bool Is_even_K, typename Params>
__forceinline__ __device__ void sparse_attn_1rowblock_sptc_int8_blockm32(
    const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;          // int8_t
    using ElementOut = typename Kernel_traits::ElementOut;     // cutlass::half_t
    using ElementAccum = typename Kernel_traits::ElementAccum; // float
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kPairBlockN = flash_sptc_int8_blockm32_impl::kSparsePairBlockN;

    // Pair kernel traits for QK^T on the pair tile (blockM x pairN)
    // Use INT8 traits for the pair path too
    using PairKernel_traits = Flash_fwd_kernel_traits_int8<
        kHeadDim, kBlockM, kPairBlockN, kNWarps,
        Kernel_traits::Is_Q_in_regs, Kernel_traits::Share_Q_K_smem>;

    const BlockInfo<!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
            cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k
                           - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // ── Global memory tensors (int8) ──────────────────────────────────────
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                    + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));

    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                    + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _),
                           Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                    + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _),
                           Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(_, 0));

    // ── Shared memory ─────────────────────────────────────────────────────
    // Normal-size smem for dense block iteration (Q + K + V)
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(),
                             typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(),
                                      typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    // ── Copy atoms ────────────────────────────────────────────────────────
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKgKBlock = tKgK(_, _, _, 0);
    auto tKgKBlockData = tKgKBlock.data();
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVgVBlock = tVgV(_, _, _, 0);
    auto tVgVBlockData = tVgVBlock.data();
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // ── MMA setup (for QK^T and dense PV) ─────────────────────────────────
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    typename Kernel_traits::TiledMmaFp16 tiled_mma_fp16;
    auto thr_mma_fp16 = tiled_mma_fp16.get_thread_slice(tidx);
    auto acc_o_layout = partition_fragment_C(tiled_mma_fp16, Shape<Int<kBlockM>, Int<kHeadDim>>{}).layout();
    constexpr int kAccOSize = decltype(size(acc_o_layout))::value;
    float acc_o_data[kAccOSize];
    auto acc_o = make_tensor(make_rmem_ptr<float>(acc_o_data), acc_o_layout);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // ── Predicates ────────────────────────────────────────────────────────
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // ── Prologue ──────────────────────────────────────────────────────────
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);
    cute::cp_async_fence();
    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    // ── Sparse block metadata ─────────────────────────────────────────────
    int num_sparse_blks = 0;
    const int* sparse_blks_ptr = nullptr;
    if (params.sparse_block_count && params.sparse_block_offset) {
        num_sparse_blks = params.sparse_block_count[(bidb * params.h + bidh) * params.NUM_ROWS + m_block];
        sparse_blks_ptr = params.sparse_block_offset +
            ((bidb * params.h + bidh) * params.NUM_ROWS + m_block) * params.NNZ_S;
    }

    int num_blks = params.block_count[(bidb * params.h + bidh) * params.NUM_ROWS + m_block];
    auto* blks_ptr = params.block_offset + ((bidb * params.h + bidh) * params.NUM_ROWS + m_block) * params.NNZ_S;
    int num_cols = params.column_count[(bidb * params.h + bidh) * params.NUM_ROWS + m_block];
    int num_cols_block = (num_cols + kBlockN - 1) / kBlockN;

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);
    flash::Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr
        ? 0.0f
        : reinterpret_cast<float*>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(
        binfo.actual_seqlen_k, binfo.actual_seqlen_q,
        params.window_size_left, params.window_size_right, alibi_slope);

    const float qk_scale = params.scale_q * params.scale_k;
    const float pv_scale = params.scale_p * params.scale_v;
    const float scale_p_inv = 1.0f / params.scale_p;

    bool is_first_softmax = true;

    // ── Sparse pair iteration (INT8 SPTC) ─────────────────────────────────
    if (num_sparse_blks > 0) {
        Tensor sQPair = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                                    typename PairKernel_traits::SmemLayoutQ{});
        Tensor sKPair = make_tensor(sQPair.data() + (PairKernel_traits::Share_Q_K_smem ? 0 : size(sQPair)),
                                    typename PairKernel_traits::SmemLayoutKV{});
        Tensor sVPair = make_tensor(sKPair.data() + size(sKPair),
                                    typename PairKernel_traits::SmemLayoutKV{});
        Tensor sKPairLeft = local_tile(sKPair, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(0, 0));
        Tensor sKPairRight = local_tile(sKPair, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(1, 0));
        Tensor sVPairLeft = local_tile(sVPair, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(0, 0));
        Tensor sVPairRight = local_tile(sVPair, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(1, 0));
        Tensor tKsKPairLeft = gmem_thr_copy_QKV.partition_D(sKPairLeft);
        Tensor tKsKPairRight = gmem_thr_copy_QKV.partition_D(sKPairRight);
        Tensor tVsVPairLeft = gmem_thr_copy_QKV.partition_D(sVPairLeft);
        Tensor tVsVPairRight = gmem_thr_copy_QKV.partition_D(sVPairRight);

        Tensor tKgK_pair = gmem_thr_copy_QKV.partition_S(gK);
        Tensor tKgKBlock_pair = tKgK_pair(_, _, _, 0);
        auto tKgKBlockData_pair = tKgKBlock_pair.data();
        Tensor tVgV_pair = gmem_thr_copy_QKV.partition_S(gV);
        Tensor tVgVBlock_pair = tVgV_pair(_, _, _, 0);
        auto tVgVBlockData_pair = tVgVBlock_pair.data();

        Tensor cKV_pair = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
        Tensor tKVcKV_pair = gmem_thr_copy_QKV.partition_S(cKV_pair);
        Tensor tKVpKV_pair = make_tensor<bool>(make_shape(size<2>(tKsKPairLeft)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tKVpKV_pair); ++k) { tKVpKV_pair(k) = get<1>(tKVcKV_pair(0, 0, k)) < params.d; }
        }

        typename PairKernel_traits::TiledMma pair_tiled_mma;
        auto pair_thr_mma = pair_tiled_mma.get_thread_slice(tidx);
        Tensor tSrKPair = pair_thr_mma.partition_fragment_B(sKPair);
        auto smem_tiled_copy_K_pair = make_tiled_copy_B(typename PairKernel_traits::SmemCopyAtom{}, pair_tiled_mma);
        auto smem_thr_copy_K_pair = smem_tiled_copy_K_pair.get_thread_slice(tidx);
        Tensor tSsKPair = smem_thr_copy_K_pair.partition_S(sKPair);

        const int row_idx_offset = m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4;
        const int warp_row_stride = kNWarps * 16;

        static_assert(kBlockM == flash_sptc_int8_blockm32_impl::kSparseBlockM, "blockm32 kernel requires kBlockM == 32.");
        static_assert(kPairBlockN == flash_sptc_int8_blockm32_impl::kSparsePairBlockN, "blockm32 kernel requires pair block-N == 64.");

        const int lane_id = tidx & 31;
        const int warp_id = tidx / 32;
        const int metadata_packed_row = warp_id * 8 + lane_id / 4;
        const uint32_t *choice_metadata_row = &sptc::kChoiceMetadataM16N8K16Packed_BandMask0_31_BlockM32_Int8[metadata_packed_row][0];
        const uint32_t mma_sp_meta = sptc::kMetadataMatrixM16N8K32_BandMask0_31_BlockM32_Int8[warp_id * 32 + lane_id][0];

        auto load_pair_half = [&](const int start_n, const bool zero_sparse_blk, auto&& tDst, const int64_t row_stride, auto&& tSrcBlock, const auto tSrcBlockData) {
            if (zero_sparse_blk) {
                cute::clear(tDst);
                return;
            }
            tSrcBlock.data() = tSrcBlockData + start_n * row_stride;
            const bool fully_valid = start_n >= 0 && start_n + kBlockN <= binfo.actual_seqlen_k;
            if (fully_valid) {
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tSrcBlock, tDst, tKVcKV_pair, tKVpKV_pair);
            } else {
                flash::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_QKV, tSrcBlock, tDst, tKVcKV_pair, tKVpKV_pair, binfo.actual_seqlen_k - start_n);
            }
        };
        auto load_pair_k = [&](const int pair_base_idx) {
            const int start_n_0 = sparse_blks_ptr[pair_base_idx - 1];
            const int start_n_1 = sparse_blks_ptr[pair_base_idx];
            const bool zero_sparse_blk_0 = (start_n_0 == -static_cast<int>(kBlockM));
            load_pair_half(start_n_0, zero_sparse_blk_0, tKsKPairLeft, int64_t(params.k_row_stride), tKgKBlock_pair, tKgKBlockData_pair);
            load_pair_half(start_n_1, false, tKsKPairRight, int64_t(params.k_row_stride), tKgKBlock_pair, tKgKBlockData_pair);
        };
        auto load_pair_v = [&](const int pair_base_idx) {
            const int start_n_0 = sparse_blks_ptr[pair_base_idx - 1];
            const int start_n_1 = sparse_blks_ptr[pair_base_idx];
            const bool zero_sparse_blk_0 = (start_n_0 == -static_cast<int>(kBlockM));
            load_pair_half(start_n_0, zero_sparse_blk_0, tVsVPairLeft, int64_t(params.v_row_stride), tVgVBlock_pair, tVgVBlockData_pair);
            load_pair_half(start_n_1, false, tVsVPairRight, int64_t(params.v_row_stride), tVgVBlock_pair, tVgVBlockData_pair);
        };

        auto apply_pair_mask = [&](auto& acc_s_pair, const int sn0, const int sn1) {
            Tensor tensor_pair = make_tensor(acc_s_pair.data(), flash::convert_layout_acc_rowcol(acc_s_pair.layout()));
            const int col_local_offset = (lane_id % 4) * 2;
            #pragma unroll
            for (int mi = 0; mi < size<0, 1>(tensor_pair); ++mi) {
                const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                #pragma unroll
                for (int i = 0; i < size<0, 0>(tensor_pair); ++i) {
                    const int row_idx = row_idx_base + i * 8;
                    const int col_idx_limit_right = Is_causal
                        ? std::min(mask.max_seqlen_k, row_idx + 1 + mask.max_seqlen_k - mask.max_seqlen_q + mask.window_size_right)
                        : mask.max_seqlen_k;
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor_pair); ++nj) {
                        const int col_local_base = col_local_offset + nj * 8;
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor_pair); ++j) {
                            const int local_col = col_local_base + j;
                            const int col_idx = (local_col < kBlockN) ? sn0 + local_col : sn1 + (local_col - kBlockN);
                            if (col_idx < 0 || col_idx >= col_idx_limit_right) {
                                tensor_pair(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                            }
                        }
                    }
                }
            }
        };

        int pair_base = num_sparse_blks - 1;
        load_pair_k(pair_base);
        cute::cp_async_fence();

        // First sparse pair: is_first_softmax=true, may need causal/local mask.
        {
            const int start_n_0 = sparse_blks_ptr[pair_base - 1];
            const int start_n_1 = sparse_blks_ptr[pair_base];
            const bool zero_sparse_blk_0 = (start_n_0 == -static_cast<int>(kBlockM));

            Tensor acc_s_pair = partition_fragment_C(pair_tiled_mma, Shape<Int<kBlockM>, Int<kPairBlockN>>{});
            clear(acc_s_pair);
            flash::cp_async_wait<0>();
            __syncthreads();

            load_pair_v(pair_base);

            flash::gemm<Kernel_traits::Is_Q_in_regs>(
                acc_s_pair, tSrQ, tSrKPair, tSsQ, tSsKPair, pair_tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K_pair,
                smem_thr_copy_Q, smem_thr_copy_K_pair);

            cute::cp_async_fence();

            // Dequant int32 → float
            constexpr int pair_acc_size = decltype(size(acc_s_pair))::value;
#ifdef FLASH_INT8_SKIP_DEQUANT
            auto acc_s_pair_f = make_tensor(make_rmem_ptr<float>(reinterpret_cast<float*>(acc_s_pair.data())), acc_s_pair.layout());
#else
            float acc_s_pair_float[pair_acc_size];
            {
                int32_t* src = reinterpret_cast<int32_t*>(acc_s_pair.data());
                #pragma unroll
                for (int i = 0; i < pair_acc_size; i++) {
                    acc_s_pair_float[i] = static_cast<float>(src[i]) * qk_scale;
                }
            }
            auto acc_s_pair_f = make_tensor(make_rmem_ptr<float>(acc_s_pair_float), acc_s_pair.layout());
#endif

            if constexpr (Is_causal || Is_local) {
                apply_pair_mask(acc_s_pair_f, start_n_0, start_n_1);
            }

            auto sp_layout = partition_fragment_C(pair_tiled_mma, Shape<Int<kBlockM>, Int<kPairBlockN / 2>>{}).layout();
            constexpr int sp_numel = decltype(size(sp_layout))::value;
            float acc_s_pair_sp_data[sp_numel];
            auto acc_s_pair_sp = make_tensor(make_rmem_ptr<float>(acc_s_pair_sp_data), sp_layout);
            flash_sptc_int8_blockm32_impl::extract_sparse_pair_int8(acc_s_pair_f, acc_s_pair_sp, choice_metadata_row, zero_sparse_blk_0);

            flash::cp_async_wait<0>();
            __syncthreads();
            if (pair_base > 1) {
                load_pair_k(pair_base - 2);
                cute::cp_async_fence();
            }
            softmax.template softmax_rescale_o<true, Is_causal || Is_local>(
                acc_s_pair_sp, acc_o, params.scale_softmax_log2);
            is_first_softmax = false;

            // Quantize sparse P float → int8 preserving CuTe fragment layout,
            // then apply convert_layout_acc_Aregs to fold N-tiles into K.
#ifdef FLASH_INT8_SKIP_DEQUANT
            Tensor rP_sp = make_tensor(make_rmem_ptr<int8_t>(acc_s_pair_sp.data()), acc_s_pair_sp.layout());
#else
            cute::array<uint32_t, sp_numel / 4> p_packed;
            {
                float* sp = acc_s_pair_sp.data();
                #pragma unroll
                for (int i = 0; i < sp_numel / 4; i++) {
                    p_packed[i] = pack_4xf32_to_s8x4(
                        sp[i * 4 + 0], sp[i * 4 + 1],
                        sp[i * 4 + 2], sp[i * 4 + 3], scale_p_inv);
                }
            }
            Tensor rP_sp = make_tensor(make_rmem_ptr<int8_t>(p_packed.data()), acc_s_pair_sp.layout());
#endif
            Tensor tOrP_sp = make_tensor(rP_sp.data(),
                flash::convert_layout_acc_Aregs<typename PairKernel_traits::TiledMma>(rP_sp.layout()));

            flash_sptc_int8_blockm32_impl::gemm_rs_sparse_mma_sp_int8<PairKernel_traits::kSwizzle, 3>(
                acc_o, tOrP_sp, reinterpret_cast<const int8_t*>(sVPair.data().get()),
                mma_sp_meta, pv_scale, kHeadDim);
            pair_base -= 2;
        }

        for (; pair_base >= 1; pair_base -= 2) {
            const int start_n_0 = sparse_blks_ptr[pair_base - 1];
            const int start_n_1 = sparse_blks_ptr[pair_base];
            const bool zero_sparse_blk_0 = (start_n_0 == -static_cast<int>(kBlockM));

            Tensor acc_s_pair = partition_fragment_C(pair_tiled_mma, Shape<Int<kBlockM>, Int<kPairBlockN>>{});
            clear(acc_s_pair);
            flash::cp_async_wait<0>();
            __syncthreads();

            load_pair_v(pair_base);

            flash::gemm<Kernel_traits::Is_Q_in_regs>(
                acc_s_pair, tSrQ, tSrKPair, tSsQ, tSsKPair, pair_tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K_pair,
                smem_thr_copy_Q, smem_thr_copy_K_pair);

            cute::cp_async_fence();

            constexpr int pair_acc_size = decltype(size(acc_s_pair))::value;
#ifdef FLASH_INT8_SKIP_DEQUANT
            auto acc_s_pair_f = make_tensor(make_rmem_ptr<float>(reinterpret_cast<float*>(acc_s_pair.data())), acc_s_pair.layout());
#else
            float acc_s_pair_float[pair_acc_size];
            {
                int32_t* src = reinterpret_cast<int32_t*>(acc_s_pair.data());
                #pragma unroll
                for (int i = 0; i < pair_acc_size; i++) {
                    acc_s_pair_float[i] = static_cast<float>(src[i]) * qk_scale;
                }
            }
            auto acc_s_pair_f = make_tensor(make_rmem_ptr<float>(acc_s_pair_float), acc_s_pair.layout());
#endif

            auto sp_layout2 = partition_fragment_C(pair_tiled_mma, Shape<Int<kBlockM>, Int<kPairBlockN / 2>>{}).layout();
            constexpr int sp_numel2 = decltype(size(sp_layout2))::value;
            float acc_s_pair_sp_data2[sp_numel2];
            auto acc_s_pair_sp = make_tensor(make_rmem_ptr<float>(acc_s_pair_sp_data2), sp_layout2);
            flash_sptc_int8_blockm32_impl::extract_sparse_pair_int8(acc_s_pair_f, acc_s_pair_sp, choice_metadata_row, zero_sparse_blk_0);

            flash::cp_async_wait<0>();
            __syncthreads();
            if (pair_base > 1) {
                load_pair_k(pair_base - 2);
                cute::cp_async_fence();
            }
            softmax.template softmax_rescale_o<false, false>(
                acc_s_pair_sp, acc_o, params.scale_softmax_log2);

#ifdef FLASH_INT8_SKIP_DEQUANT
            Tensor rP_sp2 = make_tensor(make_rmem_ptr<int8_t>(acc_s_pair_sp.data()), acc_s_pair_sp.layout());
#else
            cute::array<uint32_t, sp_numel2 / 4> p_packed2;
            {
                float* sp = acc_s_pair_sp.data();
                #pragma unroll
                for (int i = 0; i < sp_numel2 / 4; i++) {
                    p_packed2[i] = pack_4xf32_to_s8x4(
                        sp[i * 4 + 0], sp[i * 4 + 1],
                        sp[i * 4 + 2], sp[i * 4 + 3], scale_p_inv);
                }
            }
            Tensor rP_sp2 = make_tensor(make_rmem_ptr<int8_t>(p_packed2.data()), acc_s_pair_sp.layout());
#endif
            Tensor tOrP_sp2 = make_tensor(rP_sp2.data(),
                flash::convert_layout_acc_Aregs<typename PairKernel_traits::TiledMma>(rP_sp2.layout()));

            flash_sptc_int8_blockm32_impl::gemm_rs_sparse_mma_sp_int8<PairKernel_traits::kSwizzle, 3>(
                acc_o, tOrP_sp2, reinterpret_cast<const int8_t*>(sVPair.data().get()),
                mma_sp_meta, pv_scale, kHeadDim);
        }
    }

    // ── Dense block iteration (same as v1 INT8) ──────────────────────────
    if (num_blks > 0) {
        int block_index = num_blks - 1;
        tKgKBlock.data() = tKgKBlockData + blks_ptr[block_index] * int64_t(params.k_row_stride);
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgKBlock, tKsK, tKVcKV, tKVpKV,
                                           binfo.actual_seqlen_k - blks_ptr[block_index]);
        cute::cp_async_fence();

        for (; block_index >= 0; --block_index) {
            int start_n = blks_ptr[block_index];
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
            clear(acc_s);
            flash::cp_async_wait<0>();
            __syncthreads();

            tVgVBlock.data() = tVgVBlockData + start_n * int64_t(params.v_row_stride);
            flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tVgVBlock, tVsV, tKVcKV, tKVpKV);
            cute::cp_async_fence();

            flash::gemm<Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
                smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

            flash::cp_async_wait<0>();
            __syncthreads();
            if (block_index > 0) {
                tKgKBlock.data() = tKgKBlockData + blks_ptr[block_index - 1] * int64_t(params.k_row_stride);
                flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tKgKBlock, tKsK, tKVcKV, tKVpKV);
                cute::cp_async_fence();
            }

            // Dequant, mask, softmax, PV
            constexpr int acc_size = decltype(size(acc_s))::value;
#ifdef FLASH_INT8_SKIP_DEQUANT
            auto acc_s_f = make_tensor(make_rmem_ptr<float>(reinterpret_cast<float*>(acc_s.data())), acc_s.layout());
#else
            float acc_s_float[acc_size];
            int32_t* src = reinterpret_cast<int32_t*>(acc_s.data());
            #pragma unroll
            for (int i = 0; i < acc_size; i++) {
                acc_s_float[i] = static_cast<float>(src[i]) * qk_scale;
            }
            auto acc_s_f = make_tensor(make_rmem_ptr<float>(acc_s_float), acc_s.layout());
#endif

            mask.template apply_mask<Is_causal, Is_even_MN>(
                acc_s_f, start_n,
                m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16);

            if (is_first_softmax) {
                softmax.template softmax_rescale_o<true, Is_causal || Is_local>(
                    acc_s_f, acc_o, params.scale_softmax_log2);
                is_first_softmax = false;
            } else {
                softmax.template softmax_rescale_o<false, Is_causal || Is_local>(
                    acc_s_f, acc_o, params.scale_softmax_log2);
            }

            // Quantize P and dense PV
            constexpr int p_numel = decltype(size(acc_s_f))::value;
#ifdef FLASH_INT8_SKIP_DEQUANT
            Tensor rP = make_tensor(make_rmem_ptr<int8_t>(acc_s_f.data()), acc_s_f.layout());
#else
            cute::array<uint32_t, p_numel / 4> p_packed;
            float* p_data = acc_s_f.data();
            #pragma unroll
            for (int i = 0; i < p_numel / 4; i++) {
                p_packed[i] = pack_4xf32_to_s8x4(
                    p_data[i * 4 + 0], p_data[i * 4 + 1],
                    p_data[i * 4 + 2], p_data[i * 4 + 3], scale_p_inv);
            }
            Tensor rP = make_tensor(make_rmem_ptr<int8_t>(p_packed.data()), acc_s_f.layout());
#endif
            Tensor tOrP = make_tensor(rP.data(),
                flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));

            Tensor acc_pv = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
            clear(acc_pv);
            flash::gemm_rs(acc_pv, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

#ifdef FLASH_INT8_SKIP_DEQUANT
            float* pv_f = reinterpret_cast<float*>(acc_pv.data());
            float* o_data = acc_o.data();
            #pragma unroll
            for (int i = 0; i < decltype(size(acc_o))::value; i++) {
                o_data[i] += pv_f[i];
            }
#else
            int32_t* pv_data = reinterpret_cast<int32_t*>(acc_pv.data());
            float* o_data = acc_o.data();
            #pragma unroll
            for (int i = 0; i < decltype(size(acc_o))::value; i++) {
                o_data[i] += static_cast<float>(pv_data[i]) * pv_scale;
            }
#endif
        }
    }

    // ── Epilogue ──────────────────────────────────────────────────────────
    Tensor lse = softmax.template normalize_softmax_lse<false>(
        acc_o, params.scale_softmax, 1.0f);

    Tensor rO = flash::convert_type<ElementOut>(acc_o);
    Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<ElementOut*>(smem_)),
                            typename Kernel_traits::SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_fp16);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);

    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }
    cute::copy(smem_tiled_copy_O, smem_thr_copy_O.retile_S(rO), smem_thr_copy_O.partition_D(sO));

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementOut*>(params.o_ptr)
                    + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO_out = make_tensor<ElementOut>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO_out);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma_fp16.partition_C(caccO);
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    flash::copy<Is_even_MN, Is_even_K, false, false>(
        gmem_tiled_copy_O, tOrO_out, tOgO, tOcO, tOpO,
        binfo.actual_seqlen_q - m_block * kBlockM);
}

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi,
         bool Is_even_MN, bool Is_even_K, typename Params>
inline __device__ void compute_sparse_attn_sptc_int8_blockm32(const Params &params) {
    const int m_block = blockIdx.x;
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;
    flash::sparse_attn_1rowblock_sptc_int8_blockm32<Kernel_traits, Is_causal, Is_local, Has_alibi,
                                                    Is_even_MN, Is_even_K>(params, bidb, bidh, m_block);
}

} // namespace FLASH_NAMESPACE
