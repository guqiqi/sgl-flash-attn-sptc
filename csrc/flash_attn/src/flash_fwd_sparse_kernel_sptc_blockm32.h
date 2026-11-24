/******************************************************************************
 * Copyright (c) 2024, PAI, Alibaba Cloud.
 ******************************************************************************/

#pragma once

#include "flash_fwd_kernel.h"
#include "sparse_choice_metadata_bm0_31_blockm32.h"

namespace FLASH_NAMESPACE {

using namespace cute;

namespace flash_sptc_blockm32_impl {

/*
Design note for sptc sparse forward

Why sparse_block exists
- The sparse attention metadata here separates "normal blocks" from "sparse blocks".
- Normal blocks are handled the same way as v1: each selected block corresponds to one
  64 x 64 score tile, i.e. Q_tile[64, d] x K_tile[d, 64] -> score_tile[64, 64].
- sparse_block is different. It is intended for slash-pattern regions where the important
  scores are not best modeled as isolated 64 x 64 blocks. Instead, 2 sparse blocks are
  grouped into one logical pair so that the score computation is formed as a 64 x 128 tile;
  under the slash-pattern assumption, this 64 x 128 tile is then interpreted as a banded
  matrix after masking.

Logical matrix shapes
- Full attention score is:
  Q[seqlen_q, d] x K^T[d, seqlen_k] -> S[seqlen_q, seqlen_k].
- For normal block path:
  one selected block => one score fragment acc_s with shape [64, 64].
- For sparse-pair path:
  one selected pair => one score fragment acc_s_pair with shape [64, 128].
  The left 64 columns come from sparse_blks_ptr[pair_base - 1].
  The right 64 columns come from sparse_blks_ptr[pair_base].
- The pair is required because the slash pattern is treated as a banded structure inside
  this 64 x 128 score tile. Different pairs are independent in score space; they are not
  fused into one larger score fragment. However, after softmax and P x V, their output
  contributions are still accumulated into the same acc_o.

Physical K/V layout for sparse pairs
- The 2 sparse blocks of a pair are not assumed to be physically contiguous in the original
  K/V tensors.
- Therefore the sparse-pair path does not load one contiguous 128-column K/V tile from gmem.
- Instead, it materializes a logical 64 x 128 pair tile in shared memory by stitching
  together 2 separate 64-column blocks:
  left half  <- block indexed by sparse_blks_ptr[pair_base - 1]
  right half <- block indexed by sparse_blks_ptr[pair_base]
- Sparse offset sentinel: if one sparse block offset equals -64 (currently expected on the
  first sparse pair's left half), that half is treated as an all-zero 64-column block.
  We must not read K/V from gmem for that half; the corresponding shared-memory half is
  explicitly zero-filled, and score entries of that half are forced to -INFINITY before softmax.

Sparse block count contract
- num_sparse_blks is always even (processed as pairs of 2 blocks).

Band-mask semantics
- band_mask is applied on acc_s_pair as a real 64 x 128 tile.
- Its tile origin is the left block of the pair, i.e. sparse_blks_ptr[pair_base - 1].
- This means the banded geometry is defined relative to the pair's left half, while the
  right half is interpreted as columns 64..127 of the same logical tile.

Causal mask for sparse pairs
- Similar to normal blocks, sparse-pair path follows the same scheduling style:
  a masking stage then a non-masking stage.
- With current metadata ordering assumptions, the last sparse pairs are the only ones that
  can touch causal/local boundaries, so mask is only needed in the initial masking stage.

The specical compute logic for sparse blocks in sptc (using mma.sp instruction)
- The compute logic for sparse block till getting acc_s_pair is mostly the same as normal blocks.
- Before we convert the acc_s_pair type from fp32 to fp16/bf16, we need to permute the acc_s_pair layout (S[seqlen_q, seqlen_k]) along the seqlen_k dimension to prepare for the following mma.sp instruction. Here we are performing a two step permutation: 
    1. First we permute the acc_s_pair by swapping the columns ([0,127]) with the following pattern: for the i-th column, i <-> i + 63, if i < 64 and i is odd. 
        a. This way, the swapped acc_s_pair matrix conforms the required structured sparse (specifically 1:2 sparse). 
        b. And this swapped should be friendly for the kernel, considering the acc_s layout is
                mma m16n8k16 the format for C:
                -----------------------------------------------------------------
                    |     0  1     |     2  3     |     4  5     |     6  7     |
                0   | T0 :{c0, c1} | T1 :{c0, c1} | T2 :{c0, c1} | T3 :{c0, c1} |
                1   | T4 :{c0, c1} | T5 :{c0, c1} | T6 :{c0, c1} | T7 :{c0, c1} |
                ..         ..             ..             ..             ..       
                7   | T28:{c0, c1} | T29:{c0, c1} | T30:{c0, c1} | T31:{c0, c1} |
                -----------------------------------------------------------------
                8   | T0 :{c2, c3} | T1 :{c2, c3} | T2 :{c2, c3} | T3 :{c2, c3} |
                9   | T4 :{c2, c3} | T5 :{c2, c3} | T6 :{c2, c3} | T7 :{c2, c3} |
                ..         ..             ..             ..             ..       
                15  | T28:{c2, c3} | T29:{c2, c3} | T30:{c2, c3} | T31:{c2, c3} |
                -----------------------------------------------------------------
            So for each 2 swapped columns, the data are on the same threads. So it does not need to shuffle data across threads for this permutation, which should be efficient.
    2. However, this swapped acc_s_pair layout is not directly compatible with the mma.sp instruction. So we need a second permute. The mma.sp instruction requires the input matrix to be in a format:
                mma.sp m16n8k16 the format for A:
                -----------------------------------------------------------------
                    |  0  1  2  3  |  4  5  6  7  | 8  9  10  11 |12  13  14  15|
                0   | T0 :{a0, a1} | T1 :{a0, a1} | T2 :{a0, a1} | T3 :{a0, a1} |
                1   | T4 :{a0, a1} | T5 :{a0, a1} | T6 :{a0, a1} | T7 :{a0, a1} |
                ..         ..             ..             ..             ..       
                7   | T28:{a0, a1} | T29:{a0, a1} | T30:{a0, a1} | T31:{a0, a1} |
                -----------------------------------------------------------------
                8   | T0 :{a2, a3} | T1 :{a2, a3} | T2 :{a2, a3} | T3 :{a2, a3} |
                9   | T4 :{a2, a3} | T5 :{a2, a3} | T6 :{a2, a3} | T7 :{a2, a3} |
                ..         ..             ..             ..             ..       
                15  | T28:{a2, a3} | T29:{a2, a3} | T30:{a2, a3} | T31:{a2, a3} |
                -----------------------------------------------------------------
        Let's use T0 as an example, the acc_s is stored in (mma m16n8k16 the format for C), and the required mma.sp (mma.sp m16n8k16 the format for A) needs T0:{a0, a1} to be the combine of 4 values (column 0 1 and 8 9) and select only 2 values (which value to select should be computed offline).
        Considering the acc_s_pair is 1:2 structured sparsity before the second permute, this second permute result still follows 2:4 structured sparsity, satisfying the requirement of mma.sp instruction.
- For the compute logic for rP * V, the rP is permuted in the way metioned above. So the loading of V should also follow the corresponding pattern (row swap). To avoid severe overhead, this permutation should be done in loading V from shared memory to registers, instead of permuting rP in registers or permuting V in shared memory. We can do this by offseting the smem pointer when loading V to registers when calling the `ldmatrix.sync.aligned.m8n8.x4.b16`.
    1. For the first permuation step, for the i-th row, i <-> i + 63, if i < 64 and i is odd.
    2. For the second permutation step, within each 16 rows, if we want to load the i-th row of V, we need to instead i'-th row. The mapping from i to i' also follows a pattern
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] -> [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15]
        This way, the loaded V in registers should already be in the correct order for the following mma.sp instruction, without needing extra shuffle in registers or extra shared memory for permuted V.

LDMATRIX format definition used by this kernel
- CUTE atom to PTX mapping:
    - `SM75_U32x4_LDSM_N`  -> `ldmatrix.sync.aligned.x4.m8n8.shared.b16` (no transpose).
    - `SM75_U16x8_LDSM_T`  -> `ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16` (16-bit transpose).
- Warp-level contract:
    - All 32 lanes participate.
    - `.x4` loads 4 independent `8 x 8` tiles of `b16` from shared memory.
    - Each lane gets 4 x `u32` destination registers; each register packs 2 x `b16`.
- Lane/address grouping for `.x4`:
    - `lane_group = lane_id / 8` selects one of the 4 `8 x 8` tiles.
    - `lane_row = lane_id % 8` selects the row inside that tile.
    - The lane provides the shared-memory address for that `(tile, row)` pair.
- CUTE register-format reference (`Copy_Traits`):
    - For `SM75_U32x4_LDSM_N`:
      `DstLayout = Layout<Shape<_32, Shape<_32, _4>>, Stride<_32, Stride<_1, _1024>>>`.
    - For `SM75_U16x8_LDSM_T`:
      `DstLayout = Layout<Shape<Shape<_4, _8>, Shape<_16, _2, _4>>, Stride<Stride<_256, _16>, Stride<_1, _128, _1024>>>`.
- sptc implication:
    - We keep using the transposed LDSM path (`SM75_U16x8_LDSM_T`) for loading V fragments.
    - The sptc row permutation should be realized by changing the per-lane shared-memory base
      row before LDSM, so the loaded registers are already in mma.sp-consumable order.
    - No extra cross-lane shuffle is intended after LDSM.

Implementation decisions confirmed for current sptc
- PTX `mma.sp::ordered_metadata.m16n8k16` requirement:
    - Matrix A must satisfy 2:4 sparsity after permutation.
    - Metadata operand `e` is 32-bit.
    - Selector operand `f` is compile-time constant in `[0, 3]`.
    - For ordered metadata, valid nibble values are
      `{0x4, 0x8, 0x9, 0xC, 0xD, 0xE}`. (Other values are undefined behavior in PTX spec.)
- How this kernel uses metadata:
    - We precompute offline tables for `bandmask=[0,63], blockM=64` and store:
      - `kChoiceMatrixBandMask0_31_BlockM32[64][64]` (offline debug/reference table),
      - `kMetadataMatrixM16N8K16Packed_BandMask0_31_BlockM32[32][8]` (runtime table).
    - `choice` row format:
      every 4-value chunk contributes two positions `[p0, p1]` with `p in [0,3]`,
      so one 64x128 tile becomes a `64x64` choice matrix.
    - Runtime metadata table shape `[32][8]` means:
      - `32 = 4 warps * 8 lane-groups` (`lane-group = lane_id / 4`),
      - `8 = 128 / 16` k-tiles for `m16n8k16`.
      - Lanes in the same lane-group share one 32-bit packed word.
    - We pass full 32-bit `meta_packed` as operand `e`.
    - Selector `f` is warp-uniform per mma.sp call, and cycles by tile index
      (`f = tile_id % 4`, current implementation uses `tile_id = n_iter` inside each `k_iter`).
    - Lanes in the same lane-group (`lane_id / 4`) read the same metadata word.
    - In current v1+odd63+step2 build flow, only 4 nibble values appear:
      `{0x8, 0x9, 0xC, 0xD}`.
- Datatype support:
    - Both `bf16` and `fp16` paths are required for mma.sp execution.
- Mask/sparsity semantics:
    - `band_mask` + first permutation establishes the required structured sparsity.
    - `causal/local` mask can only make it "more sparse" (<= 2 non-zeros in each 4-value
      group), which remains valid for structured sparse mma.sp semantics.
- Target architecture assumption:
    - Current implementation can assume A100 execution environment; no extra non-A100
      fallback/guard is required at this stage.

    Then we need to call the mma.sp instruction as
        .reg .b32 %Ra<2>, %Rb<2>;
        .reg .f32 %Rc<4>, %Rd<4>;
        .reg .b32 %Re;
        mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {%Rd0, %Rd1, %Rd2, %Rd3},
        {%Ra0, %Ra1},
        {%Rb0, %Rb1},
        {%Rc0, %Rc1, %Rc2, %Rc3}, %Re, 0x1;

        Operands a and b represent multiplicand matrices A and B; c/d are accumulator/output.
        For structured sparse A, operand e is metadata and f is selector in [0,3]:
          f=0 -> metadata lanes T0,T4,...,T28
          f=1 -> metadata lanes T1,T5,...,T29
          f=2 -> metadata lanes T2,T6,...,T30
          f=3 -> metadata lanes T3,T7,...,T31
        It should be noted that the f for a single warp should be the same, since the mma.sp instruction requires all the threads within the warp calls the same instruction. So this f should be changed in different tiles, with tile 1 the metadata in lanes T0,T4,...,T28, tile 2 the metadata in lanes T1,T5,...,T29, tile 3 the metadata in lanes T2,T6,...,T30 and tile 4 the metadata in lanes T3,T7,...,T31, etc.
        mma.sp m16n8k16 the format for metadata:
                    col[0..3]         col[4..7]         col[8..11]        col[12..15]
        row  0   T(i)    [ 3.. 0]   T(i)    [ 7.. 4]   T(i)    [11.. 8]   T(i)    [15..12]
        row  1   T(i+4)  [ 3.. 0]   T(i+4)  [ 7.. 4]   T(i+4)  [11.. 8]   T(i+4)  [15..12]
        row  2   T(i+8)  [ 3.. 0]   T(i+8)  [ 7.. 4]   T(i+8)  [11.. 8]   T(i+8)  [15..12]
        ...
        row  7   T(i+28) [ 3.. 0]   T(i+28) [ 7.. 4]   T(i+28) [11.. 8]   T(i+28) [15..12]

        row  8   T(i)    [19..16]   T(i)    [23..20]   T(i)    [27..24]   T(i)    [31..28]
        row  9   T(i+4)  [19..16]   T(i+4)  [23..20]   T(i+4)  [27..24]   T(i+4)  [31..28]
        row 10   T(i+8)  [19..16]   T(i+8)  [23..20]   T(i+8)  [27..24]   T(i+8)  [31..28]
        ...
        row 15   T(i+28) [19..16]   T(i+28) [23..20]   T(i+28) [27..24]   T(i+28) [31..28]

- Num of blks assumption:
    Here we assume each row (each gpu block) will process at least one sparse blk pair, since the current the slashes near the main diagonal are the most important ones to capture, so we can avoid some complexity for the process of the first softmax condition. And may avoid some condition checks so the compilor can optimize better.
*/

template <typename Element, int Selector = 0>
__forceinline__ __device__ void mma_sp_m16n8k16_row_col_f32(
    float &d0, float &d1, float &d2, float &d3,
    const uint32_t a0, const uint32_t a1,
    const uint32_t b0, const uint32_t b1,
    const uint32_t meta) {
    static_assert(Selector >= 0 && Selector <= 3, "mma.sp selector must be in [0, 3].");
    if constexpr (std::is_same_v<Element, cutlass::half_t>) {
        if constexpr (Selector == 0) {
            asm volatile(
                "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, "
                "{%8, %9, %10, %11}, %12, 0x0;\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1),
                  "r"(b0), "r"(b1),
                  "f"(d0), "f"(d1), "f"(d2), "f"(d3),
                  "r"(meta));
        } else if constexpr (Selector == 1) {
            asm volatile(
                "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, "
                "{%8, %9, %10, %11}, %12, 0x1;\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1),
                  "r"(b0), "r"(b1),
                  "f"(d0), "f"(d1), "f"(d2), "f"(d3),
                  "r"(meta));
        } else if constexpr (Selector == 2) {
            asm volatile(
                "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, "
                "{%8, %9, %10, %11}, %12, 0x2;\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1),
                  "r"(b0), "r"(b1),
                  "f"(d0), "f"(d1), "f"(d2), "f"(d3),
                  "r"(meta));
        } else {
            asm volatile(
                "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, "
                "{%8, %9, %10, %11}, %12, 0x3;\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1),
                  "r"(b0), "r"(b1),
                  "f"(d0), "f"(d1), "f"(d2), "f"(d3),
                  "r"(meta));
        }
    } else if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
        if constexpr (Selector == 0) {
            asm volatile(
                "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, "
                "{%8, %9, %10, %11}, %12, 0x0;\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1),
                  "r"(b0), "r"(b1),
                  "f"(d0), "f"(d1), "f"(d2), "f"(d3),
                  "r"(meta));
        } else if constexpr (Selector == 1) {
            asm volatile(
                "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, "
                "{%8, %9, %10, %11}, %12, 0x1;\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1),
                  "r"(b0), "r"(b1),
                  "f"(d0), "f"(d1), "f"(d2), "f"(d3),
                  "r"(meta));
        } else if constexpr (Selector == 2) {
            asm volatile(
                "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, "
                "{%8, %9, %10, %11}, %12, 0x2;\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1),
                  "r"(b0), "r"(b1),
                  "f"(d0), "f"(d1), "f"(d2), "f"(d3),
                  "r"(meta));
        } else {
            asm volatile(
                "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, "
                "{%8, %9, %10, %11}, %12, 0x3;\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1),
                  "r"(b0), "r"(b1),
                  "f"(d0), "f"(d1), "f"(d2), "f"(d3),
                  "r"(meta));
        }
    } else {
        static_assert(cutlass::sizeof_bits<Element>::value == 0, "mma.sp path only supports fp16/bf16.");
    }
}

template <typename Element>
__forceinline__ __device__ void mma_sp_m16n8k16_row_col_f32_dispatch(
    float &d0, float &d1, float &d2, float &d3,
    const uint32_t a0, const uint32_t a1,
    const uint32_t b0, const uint32_t b1,
    const uint32_t meta, const int selector) {
    switch (selector) {
        case 0:
            mma_sp_m16n8k16_row_col_f32<Element, 0>(d0, d1, d2, d3, a0, a1, b0, b1, meta);
            break;
        case 1:
            mma_sp_m16n8k16_row_col_f32<Element, 1>(d0, d1, d2, d3, a0, a1, b0, b1, meta);
            break;
        case 2:
            mma_sp_m16n8k16_row_col_f32<Element, 2>(d0, d1, d2, d3, a0, a1, b0, b1, meta);
            break;
        default:
            mma_sp_m16n8k16_row_col_f32<Element, 3>(d0, d1, d2, d3, a0, a1, b0, b1, meta);
            break;
    }
}

constexpr int kSparseBlockM_SPTC_BlockM32 = 32;
constexpr int kSparsePairBlockN_SPTC_BlockM32 = 64;
constexpr int kSparseHalfBlockN_SPTC_BlockM32 = 32;
constexpr int kSparseMetadataKTiles_SPTC_BlockM32 = 4;

// Simplified extract using direct formulas derived from the two-stage permute:
//   src_t       = (dst_t & 2) | (dst_k >> 1)
//   src_k_left  = (dst_k & 1) * 2 + (dst_t & 1)   (left block, k < 4)
//   src_k_right = src_k_left + 4                    (right block, k >= 4)
// Metadata bit selects left (0) or right (1) block source.
template <int DstT, int DstK, typename TensorSrc, typename TensorDst>
__forceinline__ __device__ void extract_sparse_pair_element(
    const TensorSrc &src_fragment, TensorDst &dst_fragment,
    const uint32_t metadata_word, const int lane_group_col,
    const bool zero_left_half
) {
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
__forceinline__ __device__ void extract_sparse_pair(
    const TensorSrc &src_fragment, TensorDst &dst_fragment, const uint32_t *metadata_row,
    const bool zero_left_half = false
) {
    const int lane_group_col = threadIdx.x & 3;
#define EXTRACT_ELEMENT(DT, DK) \
    extract_sparse_pair_element<DT, DK>(src_fragment, dst_fragment, metadata_row[DK], lane_group_col, zero_left_half)
    EXTRACT_ELEMENT(0, 0); EXTRACT_ELEMENT(1, 0); EXTRACT_ELEMENT(2, 0); EXTRACT_ELEMENT(3, 0);
    EXTRACT_ELEMENT(0, 1); EXTRACT_ELEMENT(1, 1); EXTRACT_ELEMENT(2, 1); EXTRACT_ELEMENT(3, 1);
    EXTRACT_ELEMENT(0, 2); EXTRACT_ELEMENT(1, 2); EXTRACT_ELEMENT(2, 2); EXTRACT_ELEMENT(3, 2);
    EXTRACT_ELEMENT(0, 3); EXTRACT_ELEMENT(1, 3); EXTRACT_ELEMENT(2, 3); EXTRACT_ELEMENT(3, 3);
#undef EXTRACT_ELEMENT
}

template <typename Element>
__forceinline__ __device__ float unpack_b16_to_f32(uint16_t bits) {
    if constexpr (std::is_same_v<Element, cutlass::half_t>) {
        __half_raw raw;
        raw.x = bits;
        return __half2float(raw);
    } else if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        __nv_bfloat16_raw raw;
        raw.x = bits;
        return __bfloat162float(raw);
#else
        return 0.0f;
#endif
    } else {
        return 0.0f;
    }
}

template <typename Element>
__forceinline__ __device__ void debug_print_bfrag_regs(
    const char* tag,
    const int lane_id,
    const int k_iter,
    const int n_iter,
    const uint32_t* bfrag_regs,
    const int n_regs
) {
    printf("[sptc dbg] %s lane=%d k=%d n=%d regs=%d ::", tag, lane_id, k_iter, n_iter, n_regs);
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        if (r < n_regs) {
            const uint32_t reg = bfrag_regs[r];
            const uint16_t lo = static_cast<uint16_t>(reg & 0xFFFFu);
            const uint16_t hi = static_cast<uint16_t>((reg >> 16) & 0xFFFFu);
            printf(" [r%d %.0f %.0f]", r,
                   static_cast<double>(unpack_b16_to_f32<Element>(lo)),
                   static_cast<double>(unpack_b16_to_f32<Element>(hi)));
        }
    }
    printf("\n");
}

namespace sptc_blockm32_precompute {

template<int kRowSpan>
__device__ __forceinline__ int combined_v_row_map(int k_iter, int lane_row) {
    const int i = k_iter * 16 + lane_row;
    if constexpr (kRowSpan == 128) {
        // Pure bit permutation: o[6..0] = {i[0], i[5], i[4], i[1], i[3], i[2], i[6]}
        return ((i & 1) << 6) | (i & 0x30) | (((i >> 1) & 1) << 3)
             | (((i >> 3) & 1) << 2) | (((i >> 2) & 1) << 1) | ((i >> 6) & 1);
    } else if constexpr (kRowSpan == 64) {
        // Pure bit permutation: o[5..0] = {i[0], i[4], i[1], i[3], i[2], i[5]}
        return ((i & 1) << 5) | (i & 0x10) | (((i >> 1) & 1) << 3)
             | (((i >> 3) & 1) << 2) | (((i >> 2) & 1) << 1) | ((i >> 5) & 1);
    } else {
        static_assert(kRowSpan == 32, "blockm32 combined_v_row_map currently supports row spans 32, 64, or 128.");
        int row = i;
        constexpr int kHalfSpan = kRowSpan / 2;
        if (row < kHalfSpan && (row & 1)) {
            row += kHalfSpan - 1;
        } else if (row >= kHalfSpan && row <= kRowSpan - 2 && ((row & 1) == 0)) {
            row -= kHalfSpan - 1;
        }
        return map_pair_row_with_step2_m16n8k16(row);
    }
}

// Inverse of combined_v_row_map: given physical pair_row P, return the
// sequential smem row L such that combined_v_row_map maps L -> P.
// Storing V[P] at smem row kVRowInvPermute[P] lets gemm_rs_sparse_mma_sp
// read row L = k_iter*16+lane_row sequentially (no bank conflict).
__device__ constexpr int kVRowInvPermute_BlockM32[64] = {
     0, 32,  4, 36,  8, 40, 12, 44,  2, 34,  6, 38, 10, 42, 14, 46,
    16, 48, 20, 52, 24, 56, 28, 60, 18, 50, 22, 54, 26, 58, 30, 62,
     1, 33,  5, 37,  9, 41, 13, 45,  3, 35,  7, 39, 11, 43, 15, 47,
    17, 49, 21, 53, 25, 57, 29, 61, 19, 51, 23, 55, 27, 59, 31, 63,
};

} // namespace sptc_blockm32_precompute

template<typename Element, int kSwizzle, int kSwizzleS = 3,
         typename Tensor0, typename Tensor1, typename Tensor2>
__forceinline__ __device__ void gemm_rs_sparse_mma_sp(
    Tensor0 &acc, Tensor1 &rA, Tensor2 const& sB_no_swizzle,
    const uint32_t *metadata_row) {
    constexpr int kRowSpan = decltype(size<0>(sB_no_swizzle))::value;
    constexpr int kColSpan = decltype(size<1>(sB_no_swizzle))::value;
    constexpr int kKtiles = decltype(size<2>(rA))::value;
    constexpr int kNtiles = decltype(size<2>(acc))::value;

    const int lane_id = threadIdx.x & 31;
    const int lane_row = lane_id & 15;
    const int lane_col_half = lane_id >> 4;
    using ElementB = typename Tensor2::value_type;
    const ElementB* smem_base = sB_no_swizzle.data();

    static_assert(kColSpan == 64, "blockm32 pair-V path expects transposed pair block-N of 64.");
    static_assert(kSwizzleS == 4, "blockm32 pair-V path expects Swizzle<kSwizzle,3,4>.");

    constexpr int kZBase = 3;
    constexpr int kYShift = kSwizzleS;
    constexpr int kMask = (1 << kSwizzle) - 1;
    constexpr int kZMask = kMask << kZBase;
    constexpr int kYMask = kMask << (kZBase + kYShift);

    auto compute_smem_offset = [&](int row_mapped, int col_idx) __attribute__((always_inline)) {
        const int linear = (row_mapped >> 1) * kRowSpan + col_idx + (row_mapped & 1) * kRowSpan;
        const int z_bits = (linear & kZMask) >> kZBase;
        const int y_bits = (linear & kYMask) >> (kZBase + kYShift);
        return (linear & ~kZMask) | ((z_bits ^ y_bits) << kZBase);
    };

    #pragma unroll
    for (int k_iter = 0; k_iter < kKtiles; ++k_iter) {
        auto a_frag = rA(_, _, k_iter);
        auto a_regs = cute::recast<uint32_t>(a_frag);
        static_assert(decltype(size(a_regs))::value >= 2, "A fragment must provide >= 2 uint32 regs.");
        const uint32_t a0 = a_regs(0);
        const uint32_t a1 = a_regs(1);
        const int selector = k_iter & 3;
        const uint32_t meta = metadata_row[lane_id & 3];

        const int row_mapped = sptc_blockm32_precompute::combined_v_row_map<kRowSpan>(k_iter, lane_row);

        #pragma unroll
        for (int n_pair = 0; n_pair < kNtiles / 2; ++n_pair) {
            const int col_idx = n_pair * 16 + lane_col_half * 8;
            const int offset = compute_smem_offset(row_mapped, col_idx);
            const ElementB* smem_ptr = smem_base + offset;
            const uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);

            uint32_t b0, b1, b2, b3;
            asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
                : "r"(smem_int_ptr));

            {
                auto acc_tile = acc(_, _, n_pair * 2 + 0);
                CUTE_STATIC_ASSERT_V(size(acc_tile) == Int<4>{});
                mma_sp_m16n8k16_row_col_f32_dispatch<Element>(
                    acc_tile(0), acc_tile(1), acc_tile(2), acc_tile(3), a0, a1, b0, b1, meta, selector
                );
            }
            {
                auto acc_tile = acc(_, _, n_pair * 2 + 1);
                CUTE_STATIC_ASSERT_V(size(acc_tile) == Int<4>{});
                mma_sp_m16n8k16_row_col_f32_dispatch<Element>(
                    acc_tile(0), acc_tile(1), acc_tile(2), acc_tile(3), a0, a1, b2, b3, meta, selector
                );
            }
        }
    }
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
__forceinline__ __device__ void sparse_attn_1rowblock_sptc_blockm32(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    using PairKernel_traits = Flash_fwd_kernel_traits<
        kHeadDim, kBlockM, kSparsePairBlockN_SPTC_BlockM32, kNWarps,
        Kernel_traits::Is_Q_in_regs, Kernel_traits::Share_Q_K_smem, Element>;
    constexpr int kPairBlockN = PairKernel_traits::kBlockN;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("v1: m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    const index_t row_offset_k_token =
        binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
        + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v_token =
        binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
        + (bidh / params.h_h_k_ratio) * params.v_head_stride;

    Tensor gKToken = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k_token),
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_stride(_0{}, _1{}));
    Tensor gVToken = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v_token),
        Shape<Int<kBlockN>, Int<kHeadDim>>{},
        make_stride(_0{}, _1{}));
        
    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    //
    // PREDICATES
    //

    // Construct identity layout for sQ.
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
    // if (cute::thread0()) {
    //     print(tScQ.layout()); printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<0>(tScQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<1>(tScQ(i)));
    //     }
    //     printf("\n");
    // }

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
    }

    // Prologue

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);
    cute::cp_async_fence();
    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // block_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    // block_offset: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_S]
    // num_blks = tl.load(block_count + off_hz * NUM_ROWS + start_m)
    // blks_ptr = block_offset + (off_hz * NUM_ROWS + start_m) * NNZ_S
    int num_blks = params.block_count[(bidb * params.h + bidh) * params.NUM_ROWS + m_block];
    auto* blks_ptr = params.block_offset + ((bidb * params.h + bidh) * params.NUM_ROWS + m_block) * params.NNZ_S;
  
    int num_sparse_blks = params.sparse_block_count[(bidb * params.h + bidh) * params.NUM_ROWS + m_block];
    auto* sparse_blks_ptr = params.sparse_block_offset + ((bidb * params.h + bidh) * params.NUM_ROWS + m_block) * params.NNZ_S;
    // num_sparse_blks = (num_sparse_blks / 2) * 2;  // pair path requires even sparse-block count.

    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    // column_count: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M)]
    // column_index: torch.Tensor,  # [BATCH, N_HEADS, cdiv(N_CTX, BLOCK_SIZE_M), NNZ_V]
    // num_cols = tl.load(column_count + off_hz * NUM_ROWS + start_m)
    // cols_ptr = column_index + (off_hz * NUM_ROWS + start_m) * NNZ_V
    int num_cols = params.column_count[(bidb * params.h + bidh) * params.NUM_ROWS + m_block];
    int num_cols_block = (num_cols + kBlockN - 1)/ kBlockN;
    // Assumption: each row has at least one sparse block pair.
    bool is_first_softmax = true;

    if (num_sparse_blks > 0) {
        constexpr int kVSwizzleS = 4;
        using SmemLayoutAtomVPair = decltype(
            composition(Swizzle<PairKernel_traits::kSwizzle, 3, kVSwizzleS>{},
                        Layout<Shape<_8, Int<Kernel_traits::kBlockKSmem>>,
                               Stride<Int<Kernel_traits::kBlockKSmem>, _1>>{}));
        using SmemLayoutVPairReswizzled = decltype(
            tile_to_shape(SmemLayoutAtomVPair{},
                          Shape<Int<kPairBlockN>, Int<kHeadDim>>{}));
        using SmemLayoutVtPairReswizzled = decltype(
            composition(SmemLayoutVPairReswizzled{},
                        make_layout(Shape<Int<kHeadDim>, Int<kPairBlockN>>{}, GenRowMajor{})));
        using SmemLayoutVtPairReswizzledNoSwizzle = decltype(get_nonswizzle_portion(SmemLayoutVtPairReswizzled{}));
        Tensor sQPair = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                                    typename PairKernel_traits::SmemLayoutQ{});
        Tensor sKPair = make_tensor(sQPair.data() + (PairKernel_traits::Share_Q_K_smem ? 0 : size(sQPair)),
                                    typename PairKernel_traits::SmemLayoutKV{});
        Tensor sVPair = make_tensor(sKPair.data() + size(sKPair),
                                    SmemLayoutVPairReswizzled{});
        Tensor sVtNoSwizzlePair = make_tensor(sVPair.data().get(), SmemLayoutVtPairReswizzledNoSwizzle{});
        Tensor sKPairLeft = local_tile(sKPair, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(0, 0));
        Tensor sKPairRight = local_tile(sKPair, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(1, 0));
        Tensor sVPairLeft = local_tile(sVPair, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(0, 0));
        Tensor sVPairRight = local_tile(sVPair, Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_coord(1, 0));
        Tensor sVtPair = make_tensor(sVPair.data(), SmemLayoutVtPairReswizzled{});
        Tensor tKsKPairLeft = gmem_thr_copy_QKV.partition_D(sKPairLeft);
        Tensor tKsKPairRight = gmem_thr_copy_QKV.partition_D(sKPairRight);
        Tensor tVsVPairLeft = gmem_thr_copy_QKV.partition_D(sVPairLeft);
        Tensor tVsVPairRight = gmem_thr_copy_QKV.partition_D(sVPairRight);

        // Gmem source tensors for half-block loads (reuse normal block-size copy)
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
        Tensor tOrVtPair = pair_thr_mma.partition_fragment_B(sVtNoSwizzlePair);
        auto smem_tiled_copy_K_pair = make_tiled_copy_B(typename PairKernel_traits::SmemCopyAtom{}, pair_tiled_mma);
        auto smem_thr_copy_K_pair = smem_tiled_copy_K_pair.get_thread_slice(tidx);
        Tensor tSsKPair = smem_thr_copy_K_pair.partition_S(sKPair);
        auto smem_tiled_copy_V_pair = make_tiled_copy_B(typename PairKernel_traits::SmemCopyAtomTransposed{}, pair_tiled_mma);
        auto smem_thr_copy_V_pair = smem_tiled_copy_V_pair.get_thread_slice(tidx);
        Tensor tOsVtPair = smem_thr_copy_V_pair.partition_S(sVtPair);

        const int row_idx_offset = m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4;
        const int warp_row_stride = kNWarps * 16;
        constexpr int n_masking_steps_pair = (Is_causal || Is_local)
            ? ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kPairBlockN) : cute::ceil_div(kBlockM, kPairBlockN) + 1)
            : 0;
        static_assert(kBlockM == kSparseBlockM_SPTC_BlockM32, "blockm32 kernel requires kBlockM == 32.");
        static_assert(kPairBlockN == kSparsePairBlockN_SPTC_BlockM32, "blockm32 kernel requires pair block-N == 64.");
        const int lane_id = tidx & 31;
        const int warp_id = tidx / 32;
        const int metadata_packed_row = warp_id * 8 + lane_id / 4;
        const uint32_t *metadata_row = &sptc::kMetadataMatrixM16N8K16Packed_BandMask0_31_BlockM32[metadata_packed_row][0];
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
                    gmem_tiled_copy_QKV, tSrcBlock, tDst, tKVcKV_pair, tKVpKV_pair, binfo.actual_seqlen_k - start_n
                );
            }
        };
        auto load_pair_k = [&](const int pair_base_idx) {
            const int start_n_0 = sparse_blks_ptr[pair_base_idx - 1];
            const int start_n_1 = sparse_blks_ptr[pair_base_idx];
            const bool zero_sparse_blk_0 = (start_n_0 == -kSparseBlockM_SPTC_BlockM32);
            load_pair_half(start_n_0, zero_sparse_blk_0, tKsKPairLeft, int64_t(params.k_row_stride), tKgKBlock_pair, tKgKBlockData_pair);
            load_pair_half(start_n_1, /*zero_sparse_blk=*/false, tKsKPairRight, int64_t(params.k_row_stride), tKgKBlock_pair, tKgKBlockData_pair);
        };
        auto load_pair_v = [&](const int pair_base_idx) {
            const int start_n_0 = sparse_blks_ptr[pair_base_idx - 1];
            const int start_n_1 = sparse_blks_ptr[pair_base_idx];
            const bool zero_sparse_blk_0 = (start_n_0 == -kSparseBlockM_SPTC_BlockM32);
            load_pair_half(start_n_0, zero_sparse_blk_0, tVsVPairLeft, int64_t(params.v_row_stride), tVgVBlock_pair, tVgVBlockData_pair);
            load_pair_half(start_n_1, /*zero_sparse_blk=*/false, tVsVPairRight, int64_t(params.v_row_stride), tVgVBlock_pair, tVgVBlockData_pair);
        };
        int pair_base = num_sparse_blks - 1;
        load_pair_k(pair_base);
        cute::cp_async_fence();

        auto apply_pair_mask = [&](auto& acc_s_pair, const int sn0, const int sn1) {
            Tensor tensor_pair = make_tensor(acc_s_pair.data(), flash::convert_layout_acc_rowcol(acc_s_pair.layout()));
            const int lane_id = threadIdx.x % 32;
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
        auto apply_zero_half_mask = [&](auto& acc_s_pair, const bool zero_sparse_blk_0) {
            if (!zero_sparse_blk_0) return;
            Tensor tensor_pair = make_tensor(acc_s_pair.data(), flash::convert_layout_acc_rowcol(acc_s_pair.layout()));
            const int col_idx_offset = (lane_id % 4) * 2;
            #pragma unroll
            for (int mi = 0; mi < size<0, 1>(tensor_pair); ++mi) {
                #pragma unroll
                for (int i = 0; i < size<0, 0>(tensor_pair); ++i) {
                    #pragma unroll
                    for (int nj = 0; nj < size<1, 1>(tensor_pair); ++nj) {
                        const int col_idx_base = col_idx_offset + nj * 8;
                        #pragma unroll
                        for (int j = 0; j < size<1, 0>(tensor_pair); ++j) {
                            const int local_col = col_idx_base + j;
                            if (local_col < kBlockN) {
                                tensor_pair(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                            }
                        }
                    }
                }
            }
        };

        // First sparse pair: is_first_softmax=true, may need causal/local mask.
        {
            const int start_n_0 = sparse_blks_ptr[pair_base - 1];
            const int start_n_1 = sparse_blks_ptr[pair_base];
            const bool zero_sparse_blk_0 = (start_n_0 == -kSparseBlockM_SPTC_BlockM32);

            Tensor acc_s_pair = partition_fragment_C(pair_tiled_mma, Shape<Int<kBlockM>, Int<kPairBlockN>>{});
            clear(acc_s_pair);
            flash::cp_async_wait<0>();
            __syncthreads();

            // Overlap V load with gemm: sVPair and sKPair are disjoint smem regions.
            load_pair_v(pair_base);

            flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s_pair, tSrQ, tSrKPair, tSsQ, tSsKPair, pair_tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K_pair,
                smem_thr_copy_Q, smem_thr_copy_K_pair
            );

            cute::cp_async_fence();

            if constexpr (Is_softcap) {
                flash::apply_softcap(acc_s_pair, params.softcap);
            }

            if constexpr (Is_causal || Is_local) {
                apply_pair_mask(acc_s_pair, start_n_0, start_n_1);
            }

            flash::cp_async_wait<0>();
            __syncthreads();
            if (pair_base > 1) {
                load_pair_k(pair_base - 2);
                cute::cp_async_fence();
            }
            if constexpr (Is_causal || Is_local) {
                apply_zero_half_mask(acc_s_pair, zero_sparse_blk_0);
                softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/true>(
                    acc_s_pair, acc_o, params.scale_softmax_log2
                );
                is_first_softmax = false;

                Tensor rP_pair = flash::convert_type<Element>(acc_s_pair);
                Tensor tOrP_pair = make_tensor(
                    rP_pair.data(),
                    flash::convert_layout_acc_Aregs<typename PairKernel_traits::TiledMma>(rP_pair.layout())
                );
                flash::gemm_rs(
                    acc_o, tOrP_pair, tOrVtPair, tOsVtPair,
                    pair_tiled_mma, smem_tiled_copy_V_pair, smem_thr_copy_V_pair
                );
            } else {
                Tensor acc_s_pair_sp = partition_fragment_C(pair_tiled_mma, Shape<Int<kBlockM>, Int<kPairBlockN / 2>>{});
                extract_sparse_pair(acc_s_pair, acc_s_pair_sp, metadata_row, zero_sparse_blk_0);
                softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/false>(
                    acc_s_pair_sp, acc_o, params.scale_softmax_log2
                );
                is_first_softmax = false;

                Tensor rP_pair = flash::convert_type<Element>(acc_s_pair_sp);
                gemm_rs_sparse_mma_sp<Element, PairKernel_traits::kSwizzle, kVSwizzleS>(
                    acc_o, rP_pair, sVtNoSwizzlePair, metadata_row
                );
            }
            pair_base -= 2;
        }

        // Remaining sparse pairs: is_first_softmax=false, no causal/local mask needed.
        for (; pair_base >= 1; pair_base -= 2) {
            const int start_n_0 = sparse_blks_ptr[pair_base - 1];
            const bool zero_sparse_blk_0 = (start_n_0 == -kSparseBlockM_SPTC_BlockM32);

            Tensor acc_s_pair = partition_fragment_C(pair_tiled_mma, Shape<Int<kBlockM>, Int<kPairBlockN>>{});
            clear(acc_s_pair);
            flash::cp_async_wait<0>();
            __syncthreads();

            // Overlap V load with gemm: sVPair and sKPair are disjoint smem regions.
            load_pair_v(pair_base);

            flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s_pair, tSrQ, tSrKPair, tSsQ, tSsKPair, pair_tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K_pair,
                smem_thr_copy_Q, smem_thr_copy_K_pair
            );

            cute::cp_async_fence();

            if constexpr (Is_softcap) {
                flash::apply_softcap(acc_s_pair, params.softcap);
            }

            Tensor acc_s_pair_sp = partition_fragment_C(pair_tiled_mma, Shape<Int<kBlockM>, Int<kPairBlockN / 2>>{});
            extract_sparse_pair(acc_s_pair, acc_s_pair_sp, metadata_row, zero_sparse_blk_0);

            flash::cp_async_wait<0>();
            __syncthreads();
            if (pair_base > 1) {
                load_pair_k(pair_base - 2);
                cute::cp_async_fence();
            }
            if (is_first_softmax) {
                softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/false>(
                    acc_s_pair_sp, acc_o, params.scale_softmax_log2
                );
                is_first_softmax = false;
            } else {
                softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false>(
                    acc_s_pair_sp, acc_o, params.scale_softmax_log2
                );
            }

            Tensor rP_pair = flash::convert_type<Element>(acc_s_pair_sp);
            gemm_rs_sparse_mma_sp<Element, PairKernel_traits::kSwizzle, kVSwizzleS>(
                acc_o, rP_pair, sVtNoSwizzlePair, metadata_row
            );
        }
    }

    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKgKBlock = tKgK(_, _, _, 0);
    auto tKgKBlockData = tKgKBlock.data();
    Tensor tKgKToken = gmem_thr_copy_QKV.partition_S(gKToken);  // (KCPY, KCPY_N, KCPY_K)
    auto tKgKTokenData = tKgKToken.data();
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVgVBlock = tVgV(_, _, _, 0);
    auto tVgVBlockData = tVgVBlock.data();
    Tensor tVgVToken = gmem_thr_copy_QKV.partition_S(gVToken);  // (VCPY, VCPY_N, VCPY_K)
    auto tVgVTokenData = tVgVToken.data();
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    Tensor tSrK = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)
    Tensor tSgS = thr_mma.partition_C(gP);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    if (num_blks > 0) {
        int block_index = num_blks - 1;
        // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
        tKgKBlock.data() = tKgKBlockData + blks_ptr[block_index] * int64_t(params.k_row_stride);
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgKBlock, tKsK, tKVcKV, tKVpKV,
                                        binfo.actual_seqlen_k - blks_ptr[block_index]);
        cute::cp_async_fence();
        for (int n = 0; n < n_masking_steps && block_index >= 0; ++n, --block_index) {
            int start_n = blks_ptr[block_index];  // replace n_block * kBlockN

            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
            clear(acc_s);
            flash::cp_async_wait<0>();
            __syncthreads();

            // Advance gV
            tVgVBlock.data() = tVgVBlockData + start_n * int64_t(params.v_row_stride);
            if (block_index < num_blks - 1) {
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgVBlock, tVsV, tKVcKV, tKVpKV);
            } else {
                // Clear the smem tiles to account for predicated off loads
                flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                    gmem_tiled_copy_QKV, tVgVBlock, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - start_n
                );
            }
            cute::cp_async_fence();

            flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );
            // if (cute::thread0()) { print(acc_s); }
            if constexpr (Is_softcap){
                flash::apply_softcap(acc_s, params.softcap);
            }

            mask.template apply_mask<Is_causal, Is_even_MN>(
                acc_s, start_n, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
            );

            flash::cp_async_wait<0>();
            __syncthreads();
            if (block_index > 0) {
                tKgKBlock.data() = tKgKBlockData + blks_ptr[block_index - 1] * int64_t(params.k_row_stride);
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgKBlock, tKsK, tKVcKV, tKVpKV);
                // This cp_async_fence needs to be in the if block, otherwise the synchronization
                // isn't right and we get race conditions.
                cute::cp_async_fence();
            }

            // TODO: when we have key_padding_mask we'll need to Check_inf
            if (is_first_softmax) {
                softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal || Is_local>(
                    acc_s, acc_o, params.scale_softmax_log2
                );
                is_first_softmax = false;
            } else {
                softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(
                    acc_s, acc_o, params.scale_softmax_log2
                );
            }

            // Convert acc_s from fp32 to fp16/bf16
            Tensor rP = flash::convert_type<Element>(acc_s);
            int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
            int block_col_idx = n_block * (kBlockN / 32);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps
                );
                cute::copy(rP_drop, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
            }

            // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
            // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
            Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
            // if (cute::thread0()) { print(tOrP); }
            flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
            // if (cute::thread0()) { print(scores); }
        }
        for (; block_index >= 0; --block_index) {
            int start_n = blks_ptr[block_index];  // replace n_block * kBlockN

            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
            clear(acc_s);
            flash::cp_async_wait<0>();
            __syncthreads();

            // Advance gV
            tVgVBlock.data() = tVgVBlockData + start_n * int64_t(params.v_row_stride);
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgVBlock, tVsV, tKVcKV, tKVpKV);
           
            cute::cp_async_fence();

            flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );
            // if (cute::thread0()) { print(acc_s); }
            if constexpr (Is_softcap){
                flash::apply_softcap(acc_s, params.softcap);
            }

            flash::cp_async_wait<0>();
            __syncthreads();
            if (block_index > 0) {
                tKgKBlock.data() = tKgKBlockData + blks_ptr[block_index - 1] * int64_t(params.k_row_stride);
                flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgKBlock, tKsK, tKVcKV, tKVpKV);
                // This cp_async_fence needs to be in the if block, otherwise the synchronization
                // isn't right and we get race conditions.
                cute::cp_async_fence();
            }

            if (is_first_softmax) {
                softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/false>(
                    acc_s, acc_o, params.scale_softmax_log2
                );
                is_first_softmax = false;
            } else {
                softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/false>(
                    acc_s, acc_o, params.scale_softmax_log2
                );
            }

            // Convert acc_s from fp32 to fp16/bf16
            Tensor rP = flash::convert_type<Element>(acc_s);
            int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
            int block_col_idx = n_block * (kBlockN / 32);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps
                );
                cute::copy(rP_drop, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
            }

            // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
            // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
            Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
            // if (cute::thread0()) { print(tOrP); }
            flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
            // if (cute::thread0()) { print(scores); }
        }
    }
     
    if (num_cols > 0) {
        auto* cols_ptr = params.column_index + ((bidb * params.h + bidh) * params.NUM_ROWS + m_block) * params.NNZ_V;
        // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
        #pragma unroll
        for (int m = 0; m < size<1>(tKgKToken); ++m) {
            if (Is_even_MN || get<0>(tKVcKV(0, m, 0)) < num_cols) {  // Is_even_MN
                tKgKToken.data() = tKgKTokenData + cols_ptr[get<0>(tKVcKV(0, m, 0))] * int64_t(params.k_row_stride);
                #pragma unroll
                for (int k = 0; k < size<2>(tKgKToken); ++k) {
                    if (Is_even_K || tKVpKV(k)) {
                        cute::copy(gmem_tiled_copy_QKV, tKgKToken(_, m, k), tKsK(_, m, k));
                    } else {  // Clear_OOB_K
                        cute::clear(tKsK(_, m, k));
                    }
                }
            }
        }
        cute::cp_async_fence();
        for (int n = 0; n < num_cols_block; ++n) {
            // cols = tl.load(cols_ptr + start_n + offs_n, mask=n_mask, other=0)
            // int start_n = cols_ptr[n * kBlockN];  // replace n_block * kBlockN

            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
            clear(acc_s);
            flash::cp_async_wait<0>();
            __syncthreads();

            // Advance gV
            if (n < num_cols_block - 1) {
                #pragma unroll
                for (int m = 0; m < size<1>(tVgVToken); ++m) {
                    tVgVToken.data() = tVgVTokenData + cols_ptr[n * kBlockN + get<0>(tKVcKV(0, m, 0))] * int64_t(params.v_row_stride);
                    #pragma unroll
                    for (int k = 0; k < size<2>(tVgVToken); ++k) {
                        if (Is_even_K || tKVpKV(k)) {
                            cute::copy(gmem_tiled_copy_QKV, tVgVToken(_, m, k), tVsV(_, m, k));
                        } else {  // Clear_OOB_K
                            cute::clear(tVsV(_, m, k));
                        }
                    }
                }
            } else {
                // Clear the smem tiles to account for predicated off loads
                #pragma unroll
                for (int m = 0; m < size<1>(tVgVToken); ++m) {
                    if (Is_even_MN || n * kBlockN + get<0>(tKVcKV(0, m, 0)) < num_cols) {  // Is_even_MN
                        tVgVToken.data() = tVgVTokenData + cols_ptr[n * kBlockN + get<0>(tKVcKV(0, m, 0))] * int64_t(params.v_row_stride);
                        #pragma unroll
                        for (int k = 0; k < size<2>(tVgVToken); ++k) {
                            if (Is_even_K || tKVpKV(k)) {
                                cute::copy(gmem_tiled_copy_QKV, tVgVToken(_, m, k), tVsV(_, m, k));
                            } else {  // Clear_OOB_K
                                cute::clear(tVsV(_, m, k));
                            }
                        }
                    } else {  // Clear_OOB_MN
                        cute::clear(tVsV(_, m, _));
                    }
                }
            }
            cute::cp_async_fence();

            flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                smem_thr_copy_Q, smem_thr_copy_K
            );
            // if (cute::thread0()) { print(acc_s); }
            if constexpr (Is_softcap){
                flash::apply_softcap(acc_s, params.softcap);
            }

            if (n >= num_cols_block - n_masking_steps) {
                Tensor tensor = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int lane_id = threadIdx.x % 32;
                const int col_idx_offset = n * kBlockN + (lane_id % 4) * 2;
                const int row_idx_offset = m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4;
                const int warp_row_stride = kNWarps * 16;
                const int max_seqlen_k = binfo.actual_seqlen_k;
                const int max_seqlen_q = binfo.actual_seqlen_q;
                #pragma unroll
                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                    #pragma unroll
                    for (int i = 0; i < size<0, 0>(tensor); ++i) {
                        const int row_idx = row_idx_base + i * 8;
                        const int col_idx_limit = Is_causal ? std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : max_seqlen_k;
                        #pragma unroll
                        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                            const int col_idx_base = col_idx_offset + nj * 8;
                            #pragma unroll
                            for (int j = 0; j < size<1, 0>(tensor); ++j) {
                                if (col_idx_base + j >= num_cols || cols_ptr[col_idx_base + j] >= col_idx_limit) {
                                    tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                }
                            }
                        }
                    }
                }
            }

            flash::cp_async_wait<0>();
            __syncthreads();
            if (n < num_cols_block - 2) {
                #pragma unroll
                for (int m = 0; m < size<1>(tKgKToken); ++m) {
                    int token_idx = cols_ptr[(n + 1) * kBlockN + get<0>(tKVcKV(0, m, 0))];
                    tKgKToken.data() = tKgKTokenData + token_idx * int64_t(params.k_row_stride);
                    #pragma unroll
                    for (int k = 0; k < size<2>(tKgKToken); ++k) {
                        if (Is_even_K || tKVpKV(k)) {
                            cute::copy(gmem_tiled_copy_QKV, tKgKToken(_, m, k), tKsK(_, m, k));
                        } else {  // Clear_OOB_K
                            cute::clear(tKsK(_, m, k));
                        }
                    }
                }
                // This cp_async_fence needs to be in the if block, otherwise the synchronization
                // isn't right and we get race conditions.
                cute::cp_async_fence();
            } else if (n == num_cols_block - 2) {
                #pragma unroll
                for (int m = 0; m < size<1>(tKgKToken); ++m) {
                    if (Is_even_MN || (n + 1) * kBlockN + get<0>(tKVcKV(0, m, 0)) < num_cols) {  // Is_even_MN
                        int token_idx = cols_ptr[(n + 1) * kBlockN + get<0>(tKVcKV(0, m, 0))];
                        tKgKToken.data() = tKgKTokenData + token_idx * int64_t(params.k_row_stride);
                        #pragma unroll
                        for (int k = 0; k < size<2>(tKgKToken); ++k) {
                            if (Is_even_K || tKVpKV(k)) {
                                cute::copy(gmem_tiled_copy_QKV, tKgKToken(_, m, k), tKsK(_, m, k));
                            } else {  // Clear_OOB_K
                                cute::clear(tKsK(_, m, k));
                            }
                        }
                    }
                }
                cute::cp_async_fence();
            }

            // TODO: when we have key_padding_mask we'll need to Check_inf
            if (is_first_softmax) {
                softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal || Is_local>(
                    acc_s, acc_o, params.scale_softmax_log2
                );
                is_first_softmax = false;
            } else {
                softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(
                    acc_s, acc_o, params.scale_softmax_log2
                );
            }

            // Convert acc_s from fp32 to fp16/bf16
            Tensor rP = flash::convert_type<Element>(acc_s);
            int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
            int block_col_idx = n_block * (kBlockN / 32);
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                    rP_drop, block_row_idx, block_col_idx, kNWarps
                );
                cute::copy(rP_drop, tSgS);
                tSgS.data() = tSgS.data() + (-kBlockN);
            }
            if constexpr (Is_dropout) {
                dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
            }

            // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
            // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
            Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
            // if (cute::thread0()) { print(tOrP); }
            flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
            // if (cute::thread0()) { print(scores); }
        }
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_sparse_attn_sptc_blockm32(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.

    sparse_attn_1rowblock_sptc_blockm32<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block);
}

} // namespace flash_sptc_blockm32_impl

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_sparse_attn_sptc_blockm32(const Params &params) {
    flash_sptc_blockm32_impl::compute_sparse_attn_sptc_blockm32<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
}

} // namespace FLASH_NAMESPACE
