/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "namespace_config.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace FLASH_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__forceinline__ __device__ uint32_t relu2(const uint32_t x);

template<>
__forceinline__ __device__ uint32_t relu2<cutlass::half_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
#else
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela;\n" \
        "\t set.gtu.u32.f16x2 sela, %1, %2;\n" \
        "\t and.b32 %0, sela, %1;\n" 
        "}\n" : "=r"(res) : "r"(x), "r"(zero));
#endif
    return res;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<>
__forceinline__ __device__ uint32_t relu2<cutlass::bfloat16_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
    asm volatile("max.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
    return res;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

template<typename T>
__forceinline__ __device__ uint32_t convert_relu2(const float2 x);

template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::half_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::bfloat16_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator> 
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                               TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                               ThrCopy smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int kSwizzle, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs_swapped(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
                                      TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                      ThrCopy smem_thr_copy_B,
                                      Tensor4 &sB_no_swizzle) {

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    using Element = typename Tensor4::value_type;
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N

    constexpr int kStride = decltype(size<1>(sB_no_swizzle))::value;
    constexpr int kKtiles = decltype(size<2>(tCrA))::value;
    constexpr int kNtiles = decltype(size<1>(tCrB_copy_view))::value;
    constexpr int kFragRegs = decltype(size<0>(tCrB_copy_view))::value;
    constexpr bool kManualVLoad = (kStride == 128 && kKtiles == 8 && kNtiles == 8 && kFragRegs == 4);
    static_assert(!kManualVLoad, "The manual v load path is currently only implemented for a specific configuration, and needs to be updated if the configuration changes.");

    const int lane_id = threadIdx.x % 32;
    Element* smem_base = sB_no_swizzle.data();
    uint32_t b_frag[8][8][4];
    // constexpr int swap_offset[16] = {0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15};
    constexpr int swap_offset[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}; // for debug, no swap

    auto swizzle_row_col = [&](int row, int col, int &row_swz, int &col_swz) {
        constexpr int kZBase = 3;
        constexpr int kYShift = 3;
        constexpr int kMask = (1 << kSwizzle) - 1;
        constexpr int kZMask = kMask << kZBase;
        constexpr int kYMask = kMask << (kZBase + kYShift);
        int linear = row * kStride + col;
        int z_bits = (linear & kZMask) >> kZBase;
        int y_bits = (linear & kYMask) >> (kZBase + kYShift);
        int z_swz = z_bits ^ y_bits;
        int swizzled = (linear & ~kZMask) | (z_swz << kZBase);
        row_swz = swizzled / kStride;
        col_swz = swizzled - row_swz * kStride;
    };

    auto ldmatrix_trans = [&](int iter_n, int iter_k, uint32_t* dst) {
        int row_index = iter_k * 16;
        int col_index = iter_n * 16 + (lane_id / 16) * 8;
        row_index += swap_offset[lane_id % 16];

        // For step 1 swap (odd63 swap)
        // if (iter_k < 4 && lane_id % 2 == 1) {
        //     row_index += 63;
        // } else if (iter_k >= 4 && lane_id % 2 == 0) {
        //     row_index -= 63;
        // }

        int row = row_index;
        int col = col_index;
        int row_swz = 0;
        int col_swz = 0;
        swizzle_row_col(row, col, row_swz, col_swz);

        Element* smem_ptr = smem_base + row_swz * kStride / 2 + col_swz % 64 + (col_swz / 64) * (kStride / 2);
        uint32_t smem_int_ptr = cast_smem_ptr_to_uint(smem_ptr);
        asm volatile ("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
            : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
            : "r"(smem_int_ptr));
    };

    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) { // This is k tile
        cute::copy(smem_tiled_copy_B, tCsB(_, _, i), tCrB_copy_view(_, _, i));

        for (int j = 0; j < 8; ++j) { // this is n tile
            auto* dst = reinterpret_cast<uint32_t*>(b_frag[i][j]);
            ldmatrix_trans(j, i, dst);
        }

        auto a_frag = tCrA(_, _, i);
        auto a_regs = cute::recast<uint32_t>(a_frag);
        static_assert(decltype(size(a_regs))::value >= 2, "A fragment must provide at least 2 uint32 regs");
        uint32_t a0 = a_regs(0);
        uint32_t a1 = a_regs(1);
        uint32_t meta = 0;

        #pragma unroll
        for (int j = 0; j < kNtiles; ++j) {
            auto acc_tile = acc(_, _, j);
            CUTE_STATIC_ASSERT_V(size(acc_tile) == Int<4>{});
            float d0 = acc_tile(0);
            float d1 = acc_tile(1);
            float d2 = acc_tile(2);
            float d3 = acc_tile(3);

            uint32_t b0 = b_frag[i][j][0];
            uint32_t b1 = b_frag[i][j][1];
            uint32_t b2 = b_frag[i][j][2];
            uint32_t b3 = b_frag[i][j][3];

            asm volatile(
                "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, "
                "{%8, %9, %10, %11}, %12, 0;\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                : "r"(a0), "r"(a1),
                  "r"(b0), "r"(b1),
                  "f"(d0), "f"(d1), "f"(d2), "f"(d3),
                  "r"(meta));

            acc_tile(0) = d0;
            acc_tile(1) = d1;
            acc_tile(2) = d2;
            acc_tile(3) = d3;
        }
    }

    if (threadIdx.x == 0) {
        printf("[dbg] smem no-swizzle view (rows 0-127, cols 0-128)\n");
        print(sB_no_swizzle.layout());
        printf("========================================\n");
        printf("tCrA layout: ");
        print(tCrA.layout());
        printf("========================================\n");
        
        using BType = Element;
        auto unpack_16b = [](uint16_t bits) {
            if constexpr (std::is_same_v<BType, cutlass::half_t>) {
                __half_raw raw;
                raw.x = bits;
                return __half2float(raw);
            } else {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
                __nv_bfloat16_raw raw;
                raw.x = bits;
                return __bfloat162float(raw);
#else
                return 0.0f;
#endif
            }
        };

        printf("[dbg] smem swizzle view (rows 0-127, cols 0-3)\n");
        for (int row = 0; row < 128; ++row) {
            printf("  row %03d:", row);
            for (int col = 0; col < 4; ++col) {
                int row_swz = 0;
                int col_swz = 0;
                swizzle_row_col(row, col, row_swz, col_swz);
                Element* smem_ptr = smem_base + row_swz * (kStride / 2)
                    + (col_swz % 64) + (col_swz / 64) * (kStride / 2);
                uint16_t bits = reinterpret_cast<const uint16_t&>(*smem_ptr);
                printf(" %.3f", unpack_16b(bits));
            }
            printf("\n");
        }

        printf("[dbg] smem swizzled view (rows 0-127, cols 0-128)\n");
        for (int row = 0; row < 128; ++row) {
            printf("  row %03d:", row);
            for (int col = 0; col < 128; ++col) {
                int row_swz = 0;
                int col_swz = 0;
                swizzle_row_col(row, col, row_swz, col_swz);
                Element v = *(smem_base + row_swz * kStride + col_swz);
                uint16_t bits = reinterpret_cast<const uint16_t&>(v);
                printf(" %.0f", unpack_16b(bits));
            }
            printf("\n");
        }

        printf("[dbg] b_frag lane0\n");
        for (int i = 0; i < kKtiles; ++i) {
            for (int n = 0; n < kNtiles; ++n) {
                uint32_t v0 = b_frag[i][n][0];
                uint32_t v1 = b_frag[i][n][1];
                uint32_t v2 = b_frag[i][n][2];
                uint32_t v3 = b_frag[i][n][3];
                printf("  [i=%d n=%d] %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\n",
                    i, n,
                    unpack_16b(static_cast<uint16_t>(v0 & 0xFFFF)),
                    unpack_16b(static_cast<uint16_t>((v0 >> 16) & 0xFFFF)),
                    unpack_16b(static_cast<uint16_t>(v1 & 0xFFFF)),
                    unpack_16b(static_cast<uint16_t>((v1 >> 16) & 0xFFFF)),
                    unpack_16b(static_cast<uint16_t>(v2 & 0xFFFF)),
                    unpack_16b(static_cast<uint16_t>((v2 >> 16) & 0xFFFF)),
                    unpack_16b(static_cast<uint16_t>(v3 & 0xFFFF)),
                    unpack_16b(static_cast<uint16_t>((v3 >> 16) & 0xFFFF)));
            }
        }

        printf("[dbg] tCrB lane0\n");
        printf("  tCrB layout: ");
        print(tCrB.layout());
        printf("\n");
        for (int i = 0; i < kKtiles; ++i) {
            print_tensor(tCrB(_, _, i));
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to A-register layout for PV GEMM.
//   K=8  (m16n8k8):   (4, MMA_M, MMA_N)            — no change
//   K=16 (m16n8k16):  ((4, 2), MMA_M, MMA_N / 2)   — fold 2 N-tiles into K
//   K=32 (m16n8k32):  ((4, 4), MMA_M, MMA_N / 4)   — fold 4 N-tiles into K (INT8)
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16 || mma_shape_K == 32);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else if constexpr (mma_shape_K == 16) {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _4>{});  // (4, MMA_M, (4, MMA_N / 4))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_dropout(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void relu_(Tensor<Engine, Layout> &tensor) {
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
    using value_t = typename Engine::value_type;
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_uint32 = recast<uint32_t>(tensor);
    #pragma unroll
    for (int i = 0; i < size(tensor_uint32); ++i) {
        tensor_uint32(i) = relu2<value_t>(tensor_uint32(i));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// On SM80 and above, we can fuse fp32 -> fp16/bf16 conversion and relu into 1 instruction
template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type_relu(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    static_assert(std::is_same_v<To_type, cutlass::half_t> || std::is_same_v<To_type, cutlass::bfloat16_t>);
    static_assert(std::is_same_v<float, From_type>);
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_float2 = recast<float2>(tensor);
    Tensor out_uint32 = make_tensor<uint32_t>(tensor_float2.layout());
    #pragma unroll
    for (int i = 0; i < size(out_uint32); ++i) {
        out_uint32(i) = convert_relu2<To_type>(tensor_float2(i));
    }
    Tensor out = make_tensor(make_rmem_ptr<To_type>(out_uint32.data()), tensor.layout());
#else
    Tensor out = FLASH_NAMESPACE::convert_type<To_type>(tensor);
    FLASH_NAMESPACE::relu_(out);
#endif
    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
    // TD [2023-04-13]: Strange that the code below can cause race condition.
    // I think it's because the copies are under an if statement.
    // if (Is_even_K) {
    //     #pragma unroll
    //     for (int m = 0; m < size<1>(S); ++m) {
    //         if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
    //             copy(tiled_copy, S(_, m, _), D(_, m, _));
    //         } else if (Clear_OOB_MN) {
    //             clear(D(_, m, _));
    //         }
    //     }
    // } else {  // It's slightly faster in this case if iterate over K first
    //     #pragma unroll
    //     for (int k = 0; k < size<2>(S); ++k) {
    //         if (predicate_K(k)) {
    //             #pragma unroll
    //             for (int m = 0; m < size<1>(S); ++m) {
    //                 if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
    //                     copy(tiled_copy, S(_, m, k), D(_, m, k));
    //                 } else if (Clear_OOB_MN) {
    //                     clear(D(_, m, k));
    //                 }
    //             }
    //         } else if (Clear_OOB_K) {  // There's no case where !Clear_OOB_K && Clear_OOB_MN
    //             if (Clear_OOB_MN || Is_even_MN) {
    //                 clear(D(_, _, k));
    //             } else {
    //                 #pragma unroll
    //                 for (int m = 0; m < size<1>(S); ++m) {
    //                     if (!(Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN)) {
    //                         clear(D(_, m, k));
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const &S,
                                      Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                      Tensor<Engine3, Layout3> const &predicate_K,
                                      const int max_MN=0, const int min_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, max_MN = %d, min_MN = %d\n", blockIdx.y, max_MN, min_MN); }
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("Inner loop, blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_softcap(Tensor<Engine, Layout> &tensor, const float softcap){
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = cutlass::fast_tanh(tensor(i) * softcap);
    }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void calculate_dtanh(Tensor<Engine0, Layout0> &src_tensor, Tensor<Engine1, Layout1> &dst_tensor, const float softcap){
    #pragma unroll
    for (int i = 0; i < size(src_tensor); ++i) {
        dst_tensor(i) = (1.f - (src_tensor(i) * src_tensor(i))) * softcap;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace FLASH_NAMESPACE
