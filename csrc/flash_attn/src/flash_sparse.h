#pragma once
 
 #include "flash.h"
 
 namespace FLASH_NAMESPACE {
 
 struct Flash_fwd_params_sparse : public Flash_fwd_params {
     // For sparse attention
     const int* block_count;
     const int* block_offset;
     const int* column_count;
     const int* column_index;
     int NUM_ROWS;
     int NNZ_S;
     int NNZ_V;

    // For sparse attention SPTC block
    const int* sparse_block_count;
    const int* sparse_block_offset;
 };
 
 ////////////////////////////////////////////////////////////////////////////////////////////////////
 
template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_sparse_(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_sparse_sptc_(Flash_fwd_params &params, cudaStream_t stream);
template<int Headdim, bool Is_causal> void run_mha_fwd_sparse_int8_(Flash_fwd_params &params, cudaStream_t stream);
template<int Headdim, bool Is_causal> void run_mha_fwd_sparse_sptc_int8_(Flash_fwd_params &params, cudaStream_t stream);
 
 }  // namespace FLASH_NAMESPACE
