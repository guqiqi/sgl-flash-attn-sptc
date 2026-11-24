/******************************************************************************
 * INT8 kernel traits for sparse flash attention.
 *
 * Full INT8 pipeline: Q/K/V are int8, MMA uses m16n8k32 s32.s8.s8.s32,
 * accumulator is int32 for MMA and float for softmax/output.
 *
 * Output is written as fp16 (cutlass::half_t) for compatibility.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

using namespace cute;

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_>
struct Flash_kernel_traits_int8 {
    using Element = int8_t;
    using ElementOut = cutlass::half_t;   // output element type
    using ElementAccum = float;           // softmax / output accumulation
    using ElementMmaAccum = int32_t;      // MMA D/C registers
    using index_t = int64_t;
    static constexpr bool Has_cp_async = true;

    using MMA_Atom_Arch = MMA_Atom<SM80_16x8x32_S32S8S8S32_TN>;

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, int8_t>;
    // LDSM_T requires Swizzle MBase=3 which conflicts with 128-bit cp_async
    // needing MBase>=4. Use vectorized copy for transposed V load instead.
    using SmemCopyAtomTransposed = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, int8_t>;
};

template<int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_,
         bool Is_Q_in_regs_=false, bool Share_Q_K_smem_=false,
         typename Base=Flash_kernel_traits_int8<kHeadDim_, kBlockM_, kBlockN_, kNWarps_>>
struct Flash_fwd_kernel_traits_int8 : public Base {
    using Element = typename Base::Element;
    using ElementOut = typename Base::ElementOut;
    using ElementAccum = typename Base::ElementAccum;
    using ElementMmaAccum = typename Base::ElementMmaAccum;
    using index_t = typename Base::index_t;
    static constexpr bool Has_cp_async = Base::Has_cp_async;
    using SmemCopyAtom = typename Base::SmemCopyAtom;
    using SmemCopyAtomTransposed = typename Base::SmemCopyAtomTransposed;

    static constexpr bool Share_Q_K_smem = Share_Q_K_smem_;
    static constexpr bool Is_Q_in_regs = Is_Q_in_regs_ || Share_Q_K_smem;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static_assert(kHeadDim % 32 == 0);

    // INT8: 1 byte per element. Use kBlockKSmem=128 (128 bytes per row) so that
    // byte-level layout matches FP16's (64 fp16 = 128 bytes).
    // Swizzle<3,4,3>: keep bits [0,4) (16-byte blocks) intact for 128-bit cp_async
    // and ldmatrix. XOR 3 bits at [4,7) with row bits at [7,10).
    static constexpr int kBlockKSmem = kHeadDim % 128 == 0 ? 128 : 64;
    static constexpr int kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);
    static constexpr int kSwizzle = 3;

    using TiledMma = TiledMMA<
        typename Base::MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        Tile<Int<16 * kNWarps>, _16, _32>>;

    // ── SMEM layouts for Q / K / V (int8) ──────────────────────────────────
    using SmemLayoutAtomQ = decltype(
        composition(Swizzle<kSwizzle, 4, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmem>>,
                           Stride<Int<kBlockKSmem>, _1>>{}));
    using SmemLayoutQ = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));

    using SmemLayoutKV = decltype(tile_to_shape(
        SmemLayoutAtomQ{},
        Shape<Int<kBlockN>, Int<kHeadDim>>{}));

    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutKV{},
                    make_layout(Shape<Int<kHeadDim>, Int<kBlockN>>{}, GenRowMajor{})));
    using SmemLayoutVtransposedNoSwizzle = decltype(
        get_nonswizzle_portion(SmemLayoutVtransposed{}));

    // ── FP16 TiledMma for output copy tiling ──────────────────────────────
    // The output pipeline (smem_tiled_copy_C) needs a TiledMma with fp16-compatible
    // C-fragment layout. We create a dummy FP16 MMA for this purpose.
    using MMA_Atom_Fp16 = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
    using TiledMmaFp16 = TiledMMA<
        MMA_Atom_Fp16,
        Layout<Shape<Int<kNWarps>,_1,_1>>,
        Tile<Int<16 * kNWarps>, _16, _16>>;

    // ── SMEM layout for O (fp16 output) ────────────────────────────────────
    // Use fp16 sizing: kBlockKSmemO elements of fp16 = kBlockKSmemO * 2 bytes.
    // For headdim=128, fp16: kBlockKSmemO=64 → 128 bytes per row, swizzle<3,3,3>
    static constexpr int kBlockKSmemO = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kSwizzleO = kBlockKSmemO == 32 ? 2 : 3;
    using SmemLayoutAtomO = decltype(
        composition(Swizzle<kSwizzleO, 3, 3>{},
                    Layout<Shape<_8, Int<kBlockKSmemO>>,
                           Stride<Int<kBlockKSmemO>, _1>>{}));
    using SmemLayoutO = decltype(tile_to_shape(
        SmemLayoutAtomO{},
        Shape<Int<kBlockM>, Int<kHeadDim>>{}));
    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementOut>;
    using SmemCopyAtomOaccum = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>;

    // ── SMEM sizes ─────────────────────────────────────────────────────────
    static constexpr int kSmemQSize = size(SmemLayoutQ{}) * sizeof(Element);
    static constexpr int kSmemKVSize = size(SmemLayoutKV{}) * 2 * sizeof(Element);
    static constexpr int kSmemSize = Share_Q_K_smem
        ? std::max(kSmemQSize, kSmemKVSize)
        : kSmemQSize + kSmemKVSize;

    // ── Gmem copy for Q / K / V (int8, 16 elems / 128-bit) ────────────────
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);  // 16
    static_assert(kHeadDim % kGmemElemsPerLoad == 0);
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static_assert(kNThreads % kGmemThreadsPerRow == 0);
    using GmemLayoutAtom = Layout<
        Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
        Stride<Int<kGmemThreadsPerRow>, _1>>;

    using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using GmemTiledCopyQKV = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));

    // ── Gmem copy for O (fp16 output, 8 elems / 128-bit) ──────────────────
    static constexpr int kGmemElemsPerLoadO = sizeof(cute::uint128_t) / sizeof(ElementOut); // 8
    static constexpr int kGmemThreadsPerRowO = kBlockKSmemO / kGmemElemsPerLoadO;
    using GmemLayoutAtomO = Layout<
        Shape <Int<kNThreads / kGmemThreadsPerRowO>, Int<kGmemThreadsPerRowO>>,
        Stride<Int<kGmemThreadsPerRowO>, _1>>;
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementOut>{},
                        GmemLayoutAtomO{},
                        Layout<Shape<_1, _8>>{}));
};
