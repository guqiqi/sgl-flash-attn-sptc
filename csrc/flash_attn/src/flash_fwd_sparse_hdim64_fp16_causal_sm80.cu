// Copyright (c) 2023, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"
#include "namespace_config.h"
#include "flash_sparse_build_config.h"
#include "flash_fwd_sparse_launch_template.h"

namespace FLASH_NAMESPACE {

#if FLASH_SPARSE_ENABLE_V1
template<>
void run_mha_fwd_sparse_<cutlass::half_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_sparse_hdim64<cutlass::half_t, true>(params, stream);
}
#endif

#if FLASH_SPARSE_ENABLE_SPTC
template<>
void run_mha_fwd_sparse_sptc_<cutlass::half_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_sparse_hdim64_sptc<cutlass::half_t, true>(params, stream);
}
#endif

} // namespace FLASH_NAMESPACE
