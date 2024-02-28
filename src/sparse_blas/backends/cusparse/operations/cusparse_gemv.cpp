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

sycl::event optimize_gemv(sycl::queue& queue, transpose transpose_val, matrix_handle_t handle,
                          const std::vector<sycl::event>& dependencies) {
    return {};
}

template <typename fpType>
void gemv(sycl::queue& queue, transpose transpose_val, const fpType alpha, matrix_handle_t A_handle,
          sycl::buffer<fpType, 1>& x, const fpType beta, sycl::buffer<fpType, 1>& y) {}

template <typename fpType>
sycl::event gemv(sycl::queue& queue, transpose transpose_val, const fpType alpha,
                 matrix_handle_t A_handle, const fpType* x, const fpType beta, fpType* y,
                 const std::vector<sycl::event>& dependencies) {
    return {};
}

#define INSTANTIATE_GEMV(FP_TYPE, FP_SUFFIX)                                                      \
    template void gemv(sycl::queue& queue, transpose transpose_val, const FP_TYPE alpha,          \
                       matrix_handle_t A_handle, sycl::buffer<FP_TYPE, 1>& x, const FP_TYPE beta, \
                       sycl::buffer<FP_TYPE, 1>& y);                                              \
    template sycl::event gemv(sycl::queue& queue, transpose transpose_val, const FP_TYPE alpha,   \
                              matrix_handle_t A_handle, const FP_TYPE* x, const FP_TYPE beta,     \
                              FP_TYPE* y, const std::vector<sycl::event>& dependencies)

FOR_EACH_FP_TYPE(INSTANTIATE_GEMV);

#undef INSTANTIATE_GEMV

} // namespace oneapi::mkl::sparse::cusparse
