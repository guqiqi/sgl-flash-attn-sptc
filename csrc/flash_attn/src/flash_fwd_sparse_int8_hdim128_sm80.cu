// INT8 sparse attention kernel instantiation for headdim=128.
// Instantiates v1 INT8 and sptc INT8 (blockM=32/64/128) kernels.
// FP16 metadata tables are defined in the FP16 TU; only INT8 tables here.
#include "namespace_config.h"
#include "flash_sparse_build_config.h"
#define FLASH_SPTC_DEFINE_INT8_CONSTANT_TABLES
#include "flash_fwd_sparse_launch_template.h"

namespace FLASH_NAMESPACE {

#if FLASH_SPARSE_ENABLE_INT8

// v1 INT8 (dense block iteration)
template<>
void run_mha_fwd_sparse_int8_<128, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_sparse_hdim128_int8<false>(params, stream);
}
template<>
void run_mha_fwd_sparse_int8_<128, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_sparse_hdim128_int8<true>(params, stream);
}

// sptc INT8 (SPTC sparse pair path, all blockM sizes)
template<>
void run_mha_fwd_sparse_sptc_int8_<128, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_sparse_hdim128_sptc_int8<false>(params, stream);
}
template<>
void run_mha_fwd_sparse_sptc_int8_<128, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_sparse_hdim128_sptc_int8<true>(params, stream);
}

#endif // FLASH_SPARSE_ENABLE_INT8

} // namespace FLASH_NAMESPACE
