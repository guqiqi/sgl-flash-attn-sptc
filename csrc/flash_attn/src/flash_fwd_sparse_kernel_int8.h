/******************************************************************************
 * INT8 v1-style sparse attention forward kernel.
 *
 * Full INT8 pipeline: Q/K/V are int8.
 *   QK^T : CUTE TiledMma with SM80_16x8x32_S32S8S8S32_TN -> int32 acc
 *          -> dequant (scale_q * scale_k) -> float
 *   Softmax: online softmax in float (reuses softmax.h)
 *   PV : P quantize (float -> int8) -> CUTE TiledMma gemm_rs (same MMA atom)
 *        -> int32 acc -> dequant (scale_p * scale_v) -> float
 *   Output: float -> normalize -> convert to fp16 -> store
 *
 * Supports blockM = 32 (2 warps), 64 (4 warps), 128 (8 warps).
 ******************************************************************************/

#pragma once

#include "flash_fwd_kernel.h"
#include "kernel_traits_int8.h"
#include "mma_sp_int8.h"

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helpers for INT8 pipeline
////////////////////////////////////////////////////////////////////////////////////////////////////

// Dequantize int32 accumulator tensor (from QK^T MMA) to float in-place.
// combined_scale = scale_q * scale_k * scale_softmax
template<typename Tensor>
__forceinline__ __device__ void dequant_acc_inplace(Tensor &acc, float combined_scale) {
    auto* data = reinterpret_cast<int32_t*>(acc.data());
    auto* fdata = reinterpret_cast<float*>(acc.data());
    // int32 and float are both 4 bytes, can reinterpret in-place
    #pragma unroll
    for (int i = 0; i < decltype(size(acc))::value; i++) {
        fdata[i] = static_cast<float>(data[i]) * combined_scale;
    }
}

// Quantize float accumulator (softmax output P) to int8 for MMA A operand.
// Returns a tensor of int8 packed as the MMA expects.
template<typename TiledMma, typename TensorSrc>
__forceinline__ __device__ auto quantize_p_to_int8(
    const TensorSrc &p_float, float scale_p_inv) {
    constexpr int numel = decltype(size(p_float))::value;
    // Pack float -> int8 into uint32 registers (4 int8 per uint32)
    static_assert(numel % 4 == 0);
    cute::array<uint32_t, numel / 4> packed;
    const float* fptr = reinterpret_cast<const float*>(p_float.data());
    #pragma unroll
    for (int i = 0; i < numel / 4; i++) {
        packed[i] = pack_4xf32_to_s8x4(
            fptr[i * 4 + 0], fptr[i * 4 + 1],
            fptr[i * 4 + 2], fptr[i * 4 + 3], scale_p_inv);
    }
    return make_tensor(make_rmem_ptr<int8_t>(packed.data()), p_float.layout());
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Main INT8 sparse attention kernel (v1 style: dense block iteration)
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi,
         bool Is_even_MN, bool Is_even_K, typename Params>
inline __device__ void sparse_attn_1rowblock_int8(
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

    const BlockInfo<!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
            cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k
                           - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // ── Global memory tensors (Q, K, V are int8) ──────────────────────────
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

    // ── MMA setup ─────────────────────────────────────────────────────────
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    // acc_o uses FP16 MMA partition layout so it's compatible with the output copy path.
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

    // ── Prologue: load Q ──────────────────────────────────────────────────
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

    // ── Sparse block/column metadata ──────────────────────────────────────
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

    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);

    // INT8 dequant scales
    const float qk_scale = params.scale_q * params.scale_k;
    const float pv_scale = params.scale_p * params.scale_v;
    const float scale_p_inv = 1.0f / params.scale_p;

    // ── Early exit for empty sparse pattern ───────────────────────────────
    if (num_blks <= 0 && num_cols_block <= 0) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementOut*>(params.o_ptr)
                        + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                    make_shape(binfo.actual_seqlen_q, params.h, params.d),
                    make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                               make_coord(m_block, 0));
        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<ElementOut>(shape(tOgO));
        clear(tOrO);
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        flash::copy<Is_even_MN, Is_even_K, false, false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM);
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) {
                gLSE(row) = INFINITY;
            }
        }
        return;
    }

    // ── Block iteration (sparse blocks) ───────────────────────────────────
    // Lambda for one QK->softmax->PV iteration step
    auto do_qk_softmax_pv = [&](auto& acc_s_int32, int start_n, bool is_first, bool need_mask) {
        constexpr int acc_size = decltype(size(acc_s_int32))::value;

#ifdef FLASH_INT8_SKIP_DEQUANT
        // Skip dequant: reinterpret int32 bits as float directly.
        // Results are numerically meaningless but isolate dequant/quant overhead.
        auto acc_s = make_tensor(make_rmem_ptr<float>(reinterpret_cast<float*>(acc_s_int32.data())), acc_s_int32.layout());
#else
        float acc_s_float[acc_size];
        int32_t* src = reinterpret_cast<int32_t*>(acc_s_int32.data());
        #pragma unroll
        for (int i = 0; i < acc_size; i++) {
            acc_s_float[i] = static_cast<float>(src[i]) * qk_scale;
        }
        auto acc_s = make_tensor(make_rmem_ptr<float>(acc_s_float), acc_s_int32.layout());
#endif

        if (need_mask) {
            mask.template apply_mask<Is_causal, Is_even_MN>(
                acc_s, start_n,
                m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4,
                kNWarps * 16);
        }

        if (is_first) {
            softmax.template softmax_rescale_o<true, Is_causal || Is_local>(
                acc_s, acc_o, params.scale_softmax_log2);
        } else {
            softmax.template softmax_rescale_o<false, Is_causal || Is_local>(
                acc_s, acc_o, params.scale_softmax_log2);
        }

        constexpr int p_numel = decltype(size(acc_s))::value;
        static_assert(p_numel % 4 == 0);

#ifdef FLASH_INT8_SKIP_DEQUANT
        // Skip P quant: reinterpret float bits directly as int8 bytes.
        Tensor rP = make_tensor(make_rmem_ptr<int8_t>(acc_s.data()), acc_s.layout());
#else
        float* p_data = acc_s.data();
        cute::array<uint32_t, p_numel / 4> p_packed;
        #pragma unroll
        for (int i = 0; i < p_numel / 4; i++) {
            p_packed[i] = pack_4xf32_to_s8x4(
                p_data[i * 4 + 0], p_data[i * 4 + 1],
                p_data[i * 4 + 2], p_data[i * 4 + 3], scale_p_inv);
        }
        Tensor rP = make_tensor(make_rmem_ptr<int8_t>(p_packed.data()), acc_s.layout());
#endif

        Tensor tOrP = make_tensor(rP.data(),
            flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));

        Tensor acc_pv = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
        clear(acc_pv);
        flash::gemm_rs(acc_pv, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

#ifdef FLASH_INT8_SKIP_DEQUANT
        // Skip PV dequant: reinterpret int32 as float, add directly.
        float* pv_f = reinterpret_cast<float*>(acc_pv.data());
        float* o_data = acc_o.data();
        constexpr int o_size = decltype(size(acc_o))::value;
        #pragma unroll
        for (int i = 0; i < o_size; i++) {
            o_data[i] += pv_f[i];
        }
#else
        int32_t* pv_data = reinterpret_cast<int32_t*>(acc_pv.data());
        float* o_data = acc_o.data();
        constexpr int o_size = decltype(size(acc_o))::value;
        #pragma unroll
        for (int i = 0; i < o_size; i++) {
            o_data[i] += static_cast<float>(pv_data[i]) * pv_scale;
        }
#endif

    };

    if (num_blks > 0) {
        int block_index = num_blks - 1;
        tKgKBlock.data() = tKgKBlockData + blks_ptr[block_index] * int64_t(params.k_row_stride);
        flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgKBlock, tKsK, tKVcKV, tKVpKV,
                                           binfo.actual_seqlen_k - blks_ptr[block_index]);
        cute::cp_async_fence();

        for (int n = 0; n < n_masking_steps && block_index >= 0; ++n, --block_index) {
            int start_n = blks_ptr[block_index];
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
            clear(acc_s);
            flash::cp_async_wait<0>();
            __syncthreads();

            tVgVBlock.data() = tVgVBlockData + start_n * int64_t(params.v_row_stride);
            if (block_index < num_blks - 1) {
                flash::copy<true, Is_even_K>(gmem_tiled_copy_QKV, tVgVBlock, tVsV, tKVcKV, tKVpKV);
            } else {
                flash::copy<Is_even_MN, Is_even_K, true>(
                    gmem_tiled_copy_QKV, tVgVBlock, tVsV, tKVcKV, tKVpKV,
                    binfo.actual_seqlen_k - start_n);
            }
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

            do_qk_softmax_pv(acc_s, start_n, /*is_first=*/(n == 0), /*need_mask=*/true);
        }

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

            do_qk_softmax_pv(acc_s, start_n, /*is_first=*/false, /*need_mask=*/false);
        }
    }

    // ── Column iteration (vertical sparse) ────────────────────────────────
    if (num_cols > 0) {
        auto* cols_ptr = params.column_index + ((bidb * params.h + bidh) * params.NUM_ROWS + m_block) * params.NNZ_V;

        #pragma unroll
        for (int m = 0; m < size<1>(tKsK); ++m) {
            if (Is_even_MN || get<0>(tKVcKV(0, m, 0)) < num_cols) {
                const index_t row_offset_k_token =
                    binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
                    + (bidh / params.h_h_k_ratio) * params.k_head_stride;
                auto tKgKToken = make_tensor(
                    make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + row_offset_k_token
                                  + cols_ptr[get<0>(tKVcKV(0, m, 0))] * int64_t(params.k_row_stride)),
                    Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(_0{}, _1{}));
                auto tKgKTokenPart = gmem_thr_copy_QKV.partition_S(tKgKToken);
                #pragma unroll
                for (int k = 0; k < size<2>(tKsK); ++k) {
                    if (Is_even_K || tKVpKV(k)) {
                        cute::copy(gmem_tiled_copy_QKV, tKgKTokenPart(_, m, k), tKsK(_, m, k));
                    } else {
                        cute::clear(tKsK(_, m, k));
                    }
                }
            }
        }
        cute::cp_async_fence();

        for (int n = 0; n < num_cols_block; ++n) {
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
            clear(acc_s);
            flash::cp_async_wait<0>();
            __syncthreads();

            // Load V for current column block
            const index_t row_offset_v_token =
                binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)
                + (bidh / params.h_h_k_ratio) * params.v_head_stride;
            #pragma unroll
            for (int m = 0; m < size<1>(tVsV); ++m) {
                int col_idx = n * kBlockN + get<0>(tKVcKV(0, m, 0));
                if (Is_even_MN || col_idx < num_cols) {
                    auto tVgVToken = make_tensor(
                        make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr) + row_offset_v_token
                                      + cols_ptr[col_idx] * int64_t(params.v_row_stride)),
                        Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(_0{}, _1{}));
                    auto tVgVTokenPart = gmem_thr_copy_QKV.partition_S(tVgVToken);
                    #pragma unroll
                    for (int k = 0; k < size<2>(tVsV); ++k) {
                        if (Is_even_K || tKVpKV(k)) {
                            cute::copy(gmem_tiled_copy_QKV, tVgVTokenPart(_, m, k), tVsV(_, m, k));
                        } else {
                            cute::clear(tVsV(_, m, k));
                        }
                    }
                } else {
                    cute::clear(tVsV(_, m, _));
                }
            }
            cute::cp_async_fence();

            flash::gemm<Kernel_traits::Is_Q_in_regs>(
                acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma,
                smem_tiled_copy_Q, smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);

            flash::cp_async_wait<0>();
            __syncthreads();

            // Prefetch next K column block
            if (n < num_cols_block - 1) {
                #pragma unroll
                for (int m = 0; m < size<1>(tKsK); ++m) {
                    int next_col = (n + 1) * kBlockN + get<0>(tKVcKV(0, m, 0));
                    if (Is_even_MN || next_col < num_cols) {
                        auto tKgKToken = make_tensor(
                            make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)
                                + (bidh / params.h_h_k_ratio) * params.k_head_stride
                                + cols_ptr[next_col] * int64_t(params.k_row_stride)),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{}, make_stride(_0{}, _1{}));
                        auto tKgKTokenPart = gmem_thr_copy_QKV.partition_S(tKgKToken);
                        #pragma unroll
                        for (int k = 0; k < size<2>(tKsK); ++k) {
                            if (Is_even_K || tKVpKV(k)) {
                                cute::copy(gmem_tiled_copy_QKV, tKgKTokenPart(_, m, k), tKsK(_, m, k));
                            } else {
                                cute::clear(tKsK(_, m, k));
                            }
                        }
                    }
                }
                cute::cp_async_fence();
            }

            // Apply column mask
            bool need_mask = (n >= num_cols_block - n_masking_steps);
            if (need_mask) {
                Tensor tensor = make_tensor(acc_s.data(), flash::convert_layout_acc_rowcol(acc_s.layout()));
                const int lane_id = threadIdx.x % 32;
                const int col_idx_offset = n * kBlockN + (lane_id % 4) * 2;
                const int row_idx_offset = m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4;
                const int warp_row_stride = kNWarps * 16;
                #pragma unroll
                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                    const int row_idx_base = row_idx_offset + mi * warp_row_stride;
                    #pragma unroll
                    for (int i = 0; i < size<0, 0>(tensor); ++i) {
                        const int row_idx = row_idx_base + i * 8;
                        const int col_idx_limit = Is_causal
                            ? std::min(binfo.actual_seqlen_k, row_idx + 1 + binfo.actual_seqlen_k - binfo.actual_seqlen_q)
                            : binfo.actual_seqlen_k;
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

            // For column path, dequant the int32 acc_s manually since mask expects float
            // Actually acc_s from cute::gemm with int8 MMA is int32.
            // We already masked it treating int32 as float above — that won't work.
            // Need to dequant first, then mask.
            // Rework: dequant acc_s first, then mask, then softmax, then PV.
            constexpr int acc_size = decltype(size(acc_s))::value;
            float acc_s_float[acc_size];
            int32_t* src = reinterpret_cast<int32_t*>(acc_s.data());
            #pragma unroll
            for (int i = 0; i < acc_size; i++) {
                acc_s_float[i] = src[i] * qk_scale;
            }
            auto acc_s_f = make_tensor(make_rmem_ptr<float>(acc_s_float), acc_s.layout());

            // Re-apply mask on float data
            if (need_mask) {
                Tensor tensor = make_tensor(acc_s_f.data(), flash::convert_layout_acc_rowcol(acc_s_f.layout()));
                const int lane_id = threadIdx.x % 32;
                const int col_idx_offset_f = n * kBlockN + (lane_id % 4) * 2;
                const int row_idx_offset_f = m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4;
                const int warp_row_stride_f = kNWarps * 16;
                #pragma unroll
                for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
                    const int row_idx_base = row_idx_offset_f + mi * warp_row_stride_f;
                    #pragma unroll
                    for (int i = 0; i < size<0, 0>(tensor); ++i) {
                        const int row_idx = row_idx_base + i * 8;
                        const int col_idx_limit = Is_causal
                            ? std::min(binfo.actual_seqlen_k, row_idx + 1 + binfo.actual_seqlen_k - binfo.actual_seqlen_q)
                            : binfo.actual_seqlen_k;
                        #pragma unroll
                        for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                            const int col_idx_base = col_idx_offset_f + nj * 8;
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

            bool is_first = (num_blks <= 0 && n == 0);
            if (is_first) {
                softmax.template softmax_rescale_o<true, Is_causal || Is_local>(
                    acc_s_f, acc_o, params.scale_softmax_log2);
            } else {
                softmax.template softmax_rescale_o<false, Is_causal || Is_local>(
                    acc_s_f, acc_o, params.scale_softmax_log2);
            }

            // Quantize P and do PV
            constexpr int p_numel = decltype(size(acc_s_f))::value;
            cute::array<uint32_t, p_numel / 4> p_packed;
            float* p_data = acc_s_f.data();
            #pragma unroll
            for (int i = 0; i < p_numel / 4; i++) {
                p_packed[i] = pack_4xf32_to_s8x4(
                    p_data[i * 4 + 0], p_data[i * 4 + 1],
                    p_data[i * 4 + 2], p_data[i * 4 + 3], scale_p_inv);
            }
            Tensor rP = make_tensor(make_rmem_ptr<int8_t>(p_packed.data()), acc_s_f.layout());
            Tensor tOrP = make_tensor(rP.data(),
                flash::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));

            Tensor acc_pv = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
            clear(acc_pv);
            flash::gemm_rs(acc_pv, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

            int32_t* pv_data = reinterpret_cast<int32_t*>(acc_pv.data());
            float* o_data = acc_o.data();
            constexpr int o_size = decltype(size(acc_o))::value;
            #pragma unroll
            for (int i = 0; i < o_size; i++) {
                o_data[i] += static_cast<float>(pv_data[i]) * pv_scale;
            }
        }
    }

    // ── Epilogue ──────────────────────────────────────────────────────────
    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false>(
        acc_o, params.scale_softmax, /*rp_dropout=*/1.0f);

    Tensor rO = flash::convert_type<ElementOut>(acc_o);
    Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<ElementOut*>(smem_)),
                            typename Kernel_traits::SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma_fp16);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

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
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
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

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi,
         bool Is_even_MN, bool Is_even_K, typename Params>
inline __device__ void compute_sparse_attn_int8(const Params &params) {
    const int m_block = blockIdx.x;
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;
    flash::sparse_attn_1rowblock_int8<Kernel_traits, Is_causal, Is_local, Has_alibi,
                                       Is_even_MN, Is_even_K>(params, bidb, bidh, m_block);
}

} // namespace FLASH_NAMESPACE
