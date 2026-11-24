/******************************************************************************
 * Copyright (c) 2024, PAI, Alibaba Cloud.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include "flash_fwd_launch_template.h"
#include "flash_fwd_sparse_kernel.h"
#include "flash_fwd_sparse_kernel_hdim32_block64.h"
#include "flash_fwd_sparse_kernel_hdim64_block64.h"
#include "flash_fwd_sparse_kernel_sptc.h"
#include "flash_fwd_sparse_kernel_sptc_hdim32_block64.h"
#include "flash_fwd_sparse_kernel_sptc_hdim64_block64.h"
#include "flash_fwd_sparse_kernel_sptc_blockm32.h"
#include "flash_sparse_build_config.h"
#include "flash_sparse.h"
#if FLASH_SPARSE_ENABLE_INT8
#include "kernel_traits_int8.h"
#include "flash_fwd_sparse_kernel_int8.h"
#include "flash_fwd_sparse_kernel_sptc_int8_blockm32.h"
#include "flash_fwd_sparse_kernel_sptc_int8.h"
#include "flash_fwd_sparse_kernel_sptc_int8_blockm128.h"
#endif

namespace FLASH_NAMESPACE {

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_sparse_kernel, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        FLASH_NAMESPACE::compute_sparse_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_sparse_kernel_hdim32_block64, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));
        FLASH_NAMESPACE::flash_hdim32_block64_impl::compute_sparse_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_sparse_kernel_hdim64_block64, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));
        FLASH_NAMESPACE::flash_hdim64_block64_impl::compute_sparse_attn<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}



DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_sparse_kernel_sptc, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local)); // Enforce constraints
        FLASH_NAMESPACE::compute_sparse_attn_sptc<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_sparse_kernel_sptc_hdim32_block64, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));
        FLASH_NAMESPACE::flash_sptc_hdim32_block64_impl::compute_sparse_attn_sptc<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_sparse_kernel_sptc_hdim64_block64, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));
        FLASH_NAMESPACE::flash_sptc_hdim64_block64_impl::compute_sparse_attn_sptc<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_sparse_kernel_sptc_blockm32, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));
        FLASH_NAMESPACE::compute_sparse_attn_sptc_blockm32<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_sparse_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    // printf("smem_size = %d\n", smem_size);

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
            ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                    constexpr bool IsEvenMNConst = false;
                    constexpr bool Is_local = false;
                    // Will only return softmax if dropout, to reduce compilation time.
                    // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                    // If return_softmax, set IsEvenMNConst to false to reduce number of templates
                    // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                    // If Is_local, set Is_causal to false
                    auto kernel = &flash_fwd_sparse_kernel<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                    // auto kernel = &flash_fwd_kernel<Kernel_traits, false, Is_causal, false, false, true, true, false>;
                    // printf("IsEvenMNConst = %d, IsEvenKConst = %d, Is_local = %d, Is_causal = %d, ReturnSoftmaxConst = %d, Is_dropout = %d\n", int(IsEvenMNConst), int(IsEvenKConst), int(Is_local), int(Is_causal), int(ReturnSoftmaxConst), int(Is_dropout));
                    // auto kernel = &flash_fwd_kernel<Kernel_traits, false, Is_causal, false, true, true, false>;
                    if (smem_size >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    }
                    // int ctas_per_sm;
                    // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                    // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_sparse_fwd_hdim32_block64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
            ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                    constexpr bool IsEvenMNConst = false;
                    constexpr bool Is_local = false;
                    auto kernel = &flash_fwd_sparse_kernel_hdim32_block64<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                    if (smem_size >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    }
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_sparse_fwd_hdim64_block64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
            ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                    constexpr bool IsEvenMNConst = false;
                    constexpr bool Is_local = false;
                    auto kernel = &flash_fwd_sparse_kernel_hdim64_block64<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                    if (smem_size >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    }
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}



template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_sparse_fwd_sptc(Flash_fwd_params &params, cudaStream_t stream) {
    using PairKernelTraits = Flash_fwd_kernel_traits<
        Kernel_traits::kHeadDim, Kernel_traits::kBlockM, 128, Kernel_traits::kNWarps,
        Kernel_traits::Is_Q_in_regs, Kernel_traits::Share_Q_K_smem, typename Kernel_traits::Element>;
    constexpr size_t smem_size = Kernel_traits::kSmemSize > PairKernelTraits::kSmemSize
        ? Kernel_traits::kSmemSize
        : PairKernelTraits::kSmemSize;
    // printf("smem_size = %d\n", smem_size);

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
            ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                    constexpr bool IsEvenMNConst = false;
                    constexpr bool Is_local = false;
                    // Will only return softmax if dropout, to reduce compilation time.
                    // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                    // If return_softmax, set IsEvenMNConst to false to reduce number of templates
                    // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                    // If Is_local, set Is_causal to false
                    auto kernel = &flash_fwd_sparse_kernel_sptc<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                    // auto kernel = &flash_fwd_kernel<Kernel_traits, false, Is_causal, false, false, true, true, false>;
                    // printf("IsEvenMNConst = %d, IsEvenKConst = %d, Is_local = %d, Is_causal = %d, ReturnSoftmaxConst = %d, Is_dropout = %d\n", int(IsEvenMNConst), int(IsEvenKConst), int(Is_local), int(Is_causal), int(ReturnSoftmaxConst), int(Is_dropout));
                    // auto kernel = &flash_fwd_kernel<Kernel_traits, false, Is_causal, false, true, true, false>;
                    if (smem_size >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    }
                    // int ctas_per_sm;
                    // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                    // printf("smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_sparse_fwd_sptc_hdim32_block64(Flash_fwd_params &params, cudaStream_t stream) {
    using PairKernelTraits = Flash_fwd_kernel_traits<
        Kernel_traits::kHeadDim, Kernel_traits::kBlockM, 128, Kernel_traits::kNWarps,
        Kernel_traits::Is_Q_in_regs, Kernel_traits::Share_Q_K_smem, typename Kernel_traits::Element>;
    constexpr size_t smem_size = Kernel_traits::kSmemSize > PairKernelTraits::kSmemSize
        ? Kernel_traits::kSmemSize
        : PairKernelTraits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
            ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                    constexpr bool IsEvenMNConst = false;
                    constexpr bool Is_local = false;
                    auto kernel = &flash_fwd_sparse_kernel_sptc_hdim32_block64<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                    if (smem_size >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    }
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_sparse_fwd_sptc_hdim64_block64(Flash_fwd_params &params, cudaStream_t stream) {
    using PairKernelTraits = Flash_fwd_kernel_traits<
        Kernel_traits::kHeadDim, Kernel_traits::kBlockM, 128, Kernel_traits::kNWarps,
        Kernel_traits::Is_Q_in_regs, Kernel_traits::Share_Q_K_smem, typename Kernel_traits::Element>;
    constexpr size_t smem_size = Kernel_traits::kSmemSize > PairKernelTraits::kSmemSize
        ? Kernel_traits::kSmemSize
        : PairKernelTraits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
            ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                    constexpr bool IsEvenMNConst = false;
                    constexpr bool Is_local = false;
                    auto kernel = &flash_fwd_sparse_kernel_sptc_hdim64_block64<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                    if (smem_size >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    }
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_sparse_fwd_sptc_blockm32(Flash_fwd_params &params, cudaStream_t stream) {
    using PairKernelTraits = Flash_fwd_kernel_traits<
        Kernel_traits::kHeadDim, Kernel_traits::kBlockM, 64, Kernel_traits::kNWarps,
        Kernel_traits::Is_Q_in_regs, Kernel_traits::Share_Q_K_smem, typename Kernel_traits::Element>;
    constexpr size_t smem_size = Kernel_traits::kSmemSize > PairKernelTraits::kSmemSize
        ? Kernel_traits::kSmemSize
        : PairKernelTraits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
            ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
                SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                    constexpr bool IsEvenMNConst = false;
                    constexpr bool Is_local = false;
                    auto kernel = &flash_fwd_sparse_kernel_sptc_blockm32<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, Is_local && !Is_causal, Has_alibi, IsEvenMNConst && IsEvenKConst && !Is_local && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                    if (smem_size >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    }
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        TORCH_CHECK(
            params.sparse_block_size_m == 64 && params.sparse_block_size_n == 64,
            "Sparse kernels with headdim=32 currently support only block_size_m=block_size_n=64."
        );
        run_flash_sparse_fwd_hdim32_block64<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    if (params.sparse_block_size_m == 32) {
        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    } else {
        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            run_flash_sparse_fwd_hdim64_block64<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    }
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    if (params.sparse_block_size_m == 32) {
        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    } else {
        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            run_flash_sparse_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    }
}



template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim128_sptc(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    if (params.sparse_block_size_m == 32) {
        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            run_flash_sparse_fwd_sptc_blockm32<Flash_fwd_kernel_traits<Headdim, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    } else {
        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            run_flash_sparse_fwd_sptc<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    }
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim32_sptc(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        TORCH_CHECK(
            params.sparse_block_size_m == 64 && params.sparse_block_size_n == 64,
            "Sparse sptc with headdim=32 currently supports only block_size_m=block_size_n=64."
        );
        run_flash_sparse_fwd_sptc_hdim32_block64<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_sparse_hdim64_sptc(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    if (params.sparse_block_size_m == 32) {
        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            run_flash_sparse_fwd_sptc_blockm32<Flash_fwd_kernel_traits<Headdim, 32, 32, 2, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    } else {
        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            run_flash_sparse_fwd_sptc_hdim64_block64<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        });
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// INT8 kernel definitions and launch functions
////////////////////////////////////////////////////////////////////////////////////////////////////

#if FLASH_SPARSE_ENABLE_INT8

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_sparse_kernel_int8, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));
        FLASH_NAMESPACE::compute_sparse_attn_int8<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_sparse_kernel_sptc_int8_blockm32, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));
        FLASH_NAMESPACE::compute_sparse_attn_sptc_int8_blockm32<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_sparse_kernel_sptc_int8, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));
        FLASH_NAMESPACE::compute_sparse_attn_sptc_int8<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_sparse_kernel_sptc_int8_blockm128, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K) {
    #if defined(ARCH_SUPPORTS_FLASH)
        static_assert(!(Is_causal && Is_local));
        FLASH_NAMESPACE::compute_sparse_attn_sptc_int8_blockm128<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

// INT8 v1 launch function
template<typename Kernel_traits, bool Is_causal>
void run_flash_sparse_fwd_int8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
            constexpr bool IsEvenMNConst = false;
            constexpr bool Is_local = false;
            auto kernel = &flash_fwd_sparse_kernel_int8<Kernel_traits, Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst>;
            if (smem_size >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            }
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

// INT8 sptc blockm32 launch function
template<typename Kernel_traits, bool Is_causal>
void run_flash_sparse_fwd_sptc_int8_blockm32(Flash_fwd_params &params, cudaStream_t stream) {
    using PairKernelTraits = Flash_fwd_kernel_traits_int8<
        Kernel_traits::kHeadDim, Kernel_traits::kBlockM, 64, Kernel_traits::kNWarps,
        Kernel_traits::Is_Q_in_regs, Kernel_traits::Share_Q_K_smem>;
    constexpr size_t smem_size = Kernel_traits::kSmemSize > PairKernelTraits::kSmemSize
        ? Kernel_traits::kSmemSize : PairKernelTraits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
            constexpr bool IsEvenMNConst = false;
            constexpr bool Is_local = false;
            auto kernel = &flash_fwd_sparse_kernel_sptc_int8_blockm32<Kernel_traits, Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst>;
            if (smem_size >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            }
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

// INT8 sptc blockm64 launch function
template<typename Kernel_traits, bool Is_causal>
void run_flash_sparse_fwd_sptc_int8(Flash_fwd_params &params, cudaStream_t stream) {
    using PairKernelTraits = Flash_fwd_kernel_traits_int8<
        Kernel_traits::kHeadDim, Kernel_traits::kBlockM, 128, Kernel_traits::kNWarps,
        Kernel_traits::Is_Q_in_regs, Kernel_traits::Share_Q_K_smem>;
    constexpr size_t smem_size = Kernel_traits::kSmemSize > PairKernelTraits::kSmemSize
        ? Kernel_traits::kSmemSize : PairKernelTraits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
            constexpr bool IsEvenMNConst = false;
            constexpr bool Is_local = false;
            auto kernel = &flash_fwd_sparse_kernel_sptc_int8<Kernel_traits, Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst>;
            if (smem_size >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            }
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

// INT8 sptc blockm128 launch function
template<typename Kernel_traits, bool Is_causal>
void run_flash_sparse_fwd_sptc_int8_blockm128(Flash_fwd_params &params, cudaStream_t stream) {
    using PairKernelTraits = Flash_fwd_kernel_traits_int8<
        Kernel_traits::kHeadDim, Kernel_traits::kBlockM, 256, Kernel_traits::kNWarps,
        Kernel_traits::Is_Q_in_regs, Kernel_traits::Share_Q_K_smem>;
    constexpr size_t smem_size = Kernel_traits::kSmemSize > PairKernelTraits::kSmemSize
        ? Kernel_traits::kSmemSize : PairKernelTraits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        ALIBI_SWITCH(params.alibi_slopes_ptr != nullptr, Has_alibi, [&] {
            constexpr bool IsEvenMNConst = false;
            constexpr bool Is_local = false;
            auto kernel = &flash_fwd_sparse_kernel_sptc_int8_blockm128<Kernel_traits, Is_causal, Is_local, Has_alibi, IsEvenMNConst, IsEvenKConst>;
            if (smem_size >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            }
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

// INT8 top-level dispatchers
template<bool Is_causal>
void run_mha_fwd_sparse_hdim128_int8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    if (params.sparse_block_size_m == 32) {
        run_flash_sparse_fwd_int8<Flash_fwd_kernel_traits_int8<Headdim, 32, 32, 2, false, false>, Is_causal>(params, stream);
    } else if (params.sparse_block_size_m == 128) {
        run_flash_sparse_fwd_int8<Flash_fwd_kernel_traits_int8<Headdim, 128, 128, 8, false, false>, Is_causal>(params, stream);
    } else {
        run_flash_sparse_fwd_int8<Flash_fwd_kernel_traits_int8<Headdim, 64, 64, 4, false, false>, Is_causal>(params, stream);
    }
}

template<bool Is_causal>
void run_mha_fwd_sparse_hdim128_sptc_int8(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    if (params.sparse_block_size_m == 32) {
        run_flash_sparse_fwd_sptc_int8_blockm32<Flash_fwd_kernel_traits_int8<Headdim, 32, 32, 2, false, false>, Is_causal>(params, stream);
    } else if (params.sparse_block_size_m == 128) {
        run_flash_sparse_fwd_sptc_int8_blockm128<Flash_fwd_kernel_traits_int8<Headdim, 128, 128, 8, false, false>, Is_causal>(params, stream);
    } else {
        run_flash_sparse_fwd_sptc_int8<Flash_fwd_kernel_traits_int8<Headdim, 64, 64, 4, false, false>, Is_causal>(params, stream);
    }
}

#endif // FLASH_SPARSE_ENABLE_INT8

} // namespace FLASH_NAMESPACE
