#pragma once

#include "flash_fwd_kernel.h"

namespace FLASH_NAMESPACE {

using namespace cute;

// Debug helpers to validate mask logic in CUDA.
// They build an accumulator fragment with small (0-9) indices, apply the same masks,
// and print the final kBlockM x kBlockN tile.

template <typename Kernel_traits, typename AccTensor>
inline __device__ void sparse_debug_fill_acc_with_indices(AccTensor &acc, const int tidx) {
    // Fill the per-thread MMA accumulator fragment with a deterministic 0-9 pattern,
    // addressing it through a row/col-friendly view.
    Tensor tensor_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));

    const int lane_id = tidx % 32;
    const int warp_id = tidx / 32;
    const int row_in_tile_base = warp_id * 16 + lane_id / 4;
    const int col_in_tile_base = (lane_id % 4) * 2;

    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor_rc); ++mi) {
        const int row_in_tile_mi = row_in_tile_base + mi * (Kernel_traits::kNWarps * 16);
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor_rc); ++i) {
            const int row_in_tile = row_in_tile_mi + i * 8;
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor_rc); ++nj) {
                const int col_in_tile_nj = col_in_tile_base + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor_rc); ++j) {
                    const int col_in_tile = col_in_tile_nj + j;
                    if (row_in_tile < Kernel_traits::kBlockM && col_in_tile < Kernel_traits::kBlockN) {
                        const int idx = row_in_tile * Kernel_traits::kBlockN + col_in_tile;
                        tensor_rc(make_coord(i, mi), make_coord(j, nj)) = static_cast<float>(idx % 10);
                    }
                }
            }
        }
    }
}

template <typename Kernel_traits, typename AccTensor>
inline __device__ void sparse_debug_print_acc_tile(
    const AccTensor &acc,
    const int tidx,
    const int bidb,
    const int bidh,
    const int m_block,
    const int start_n,
    const int n_iter,
    const int stage
) {
    __shared__ float dbg_tile_smem[Kernel_traits::kBlockM * Kernel_traits::kBlockN];

    // Initialize to NaN sentinel so we can tell if any element wasn't written.
    for (int idx = tidx; idx < Kernel_traits::kBlockM * Kernel_traits::kBlockN; idx += blockDim.x) {
        dbg_tile_smem[idx] = __int_as_float(0x7fffffff);
    }
    __syncthreads();

    Tensor tensor_rc = make_tensor(acc.data(), flash::convert_layout_acc_rowcol(acc.layout()));
    const int lane_id = tidx % 32;
    const int warp_id = tidx / 32;
    const int row_in_tile_base = warp_id * 16 + lane_id / 4;
    const int col_in_tile_base = (lane_id % 4) * 2;

    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor_rc); ++mi) {
        const int row_in_tile_mi = row_in_tile_base + mi * (Kernel_traits::kNWarps * 16);
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor_rc); ++i) {
            const int row_in_tile = row_in_tile_mi + i * 8;
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor_rc); ++nj) {
                const int col_in_tile_nj = col_in_tile_base + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor_rc); ++j) {
                    const int col_in_tile = col_in_tile_nj + j;
                    if (row_in_tile < Kernel_traits::kBlockM && col_in_tile < Kernel_traits::kBlockN) {
                        dbg_tile_smem[row_in_tile * Kernel_traits::kBlockN + col_in_tile] =
                            static_cast<float>(tensor_rc(make_coord(i, mi), make_coord(j, nj)));
                    }
                }
            }
        }
    }
    __syncthreads();

    if (tidx == 0) {
        const char* stage_str = stage == 0 ? "fill" : (stage == 1 ? "masked" : "?");
        printf("[FLASH_SPARSE_DEBUG_MASK_TILE] stage=%s bidb=%d bidh=%d m_block=%d start_n=%d n_iter=%d kBlockM=%d kBlockN=%d\n",
               stage_str, bidb, bidh, m_block, start_n, n_iter, Kernel_traits::kBlockM, Kernel_traits::kBlockN);
        // int count = 0;
        for (int rm = 0; rm < Kernel_traits::kBlockM; ++rm) {
        // for (int rm = 0; rm < 16; ++rm) {
            printf("row %02d: ", rm);
            for (int cn = 0; cn < Kernel_traits::kBlockN; ++cn) {
                const float v = dbg_tile_smem[rm * Kernel_traits::kBlockN + cn];
                // Masked regions should be -inf.
                if (v <= -1.0e20f) {
                    printf(" .");
                } else if (v != v) {
                    // Not written sentinel / NaN
                    printf(" ?");
                } else {
                    const int digit = static_cast<int>(v + 0.5f);
                    printf(" %d", digit);
                    // printf(" %.3f", v);
                    // count++;
                }
                // if (cn % 4 == 3 && cn > 0) {
                //     printf(" |");
                //     if (count > 2) printf("Error! Voliating 2:4 Sparisty");
                //     count = 0;
                // }
            }
            printf("\n");
        }
    }
    __syncthreads();
}

} // namespace FLASH_NAMESPACE