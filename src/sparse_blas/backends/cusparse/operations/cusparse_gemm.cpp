/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

#include "oneapi/mkl/sparse_blas/detail/cusparse/onemkl_sparse_blas_cusparse.hpp"

#include "sparse_blas/backends/cusparse/cusparse_error.hpp"
#include "sparse_blas/backends/cusparse/cusparse_helper.hpp"
#include "sparse_blas/backends/cusparse/cusparse_task.hpp"
#include "sparse_blas/backends/cusparse/cusparse_internal_containers.hpp"
#include "sparse_blas/macros.hpp"

namespace oneapi::mkl::sparse::cusparse {

// TODO(Romain): Update to new API

sycl::event optimize_gemm(sycl::queue& queue, transpose transpose_A, matrix_handle_t handle,
                          const std::vector<sycl::event>& dependencies) {
    return {};
}

sycl::event optimize_gemm(sycl::queue& queue, transpose transpose_A, transpose transpose_B,
                          layout dense_matrix_layout, const std::int64_t columns,
                          matrix_handle_t handle, const std::vector<sycl::event>& dependencies) {
    return {};
}

template <typename fpType>
void gemm(sycl::queue& queue, layout dense_matrix_layout, transpose transpose_A,
          transpose transpose_B, const fpType alpha, matrix_handle_t A_handle,
          sycl::buffer<fpType, 1>& B, const std::int64_t columns, const std::int64_t ldb,
          const fpType beta, sycl::buffer<fpType, 1>& C, const std::int64_t ldc) {}

template <typename fpType>
sycl::event gemm(sycl::queue& queue, layout dense_matrix_layout, transpose transpose_A,
                 transpose transpose_B, const fpType alpha, matrix_handle_t A_handle,
                 const fpType* B, const std::int64_t columns, const std::int64_t ldb,
                 const fpType beta, fpType* C, const std::int64_t ldc,
                 const std::vector<sycl::event>& dependencies) {
    return {};
}

#define INSTANTIATE_GEMM(FP_TYPE, FP_SUFFIX)                                                    \
    template void gemm(sycl::queue& queue, layout dense_matrix_layout, transpose transpose_A,   \
                       transpose transpose_B, const FP_TYPE alpha, matrix_handle_t A_handle,    \
                       sycl::buffer<FP_TYPE, 1>& B, const std::int64_t columns,                 \
                       const std::int64_t ldb, const FP_TYPE beta, sycl::buffer<FP_TYPE, 1>& C, \
                       const std::int64_t ldc);                                                 \
    template sycl::event gemm(                                                                  \
        sycl::queue& queue, layout dense_matrix_layout, transpose transpose_A,                  \
        transpose transpose_B, const FP_TYPE alpha, matrix_handle_t A_handle, const FP_TYPE* B, \
        const std::int64_t columns, const std::int64_t ldb, const FP_TYPE beta, FP_TYPE* C,     \
        const std::int64_t ldc, const std::vector<sycl::event>& dependencies)

FOR_EACH_FP_TYPE(INSTANTIATE_GEMM);

#undef INSTANTIATE_GEMM

} // namespace oneapi::mkl::sparse::cusparse
