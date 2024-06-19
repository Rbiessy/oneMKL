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

#ifndef _ONEMKL_SRC_SPARSE_BLAS_COMMON_OP_VERIFICATION_HPP_
#define _ONEMKL_SRC_SPARSE_BLAS_COMMON_OP_VERIFICATION_HPP_

#include <string>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl/sparse_blas/types.hpp"
#include "macros.hpp"

namespace oneapi::mkl::sparse::detail {

/// Return whether a pointer is accessible on the host
template <typename T>
inline bool is_ptr_accessible_on_host(sycl::queue &queue, const T *host_or_device_ptr) {
    auto alloc_type = sycl::get_pointer_type(host_or_device_ptr, queue.get_context());
    // Note sycl::usm::alloc::host may not be accessible on the host according to SYCL specification.
    // sycl::usm::alloc::unknown is returned if the pointer is not a USM allocation which is assumed to be a normal host pointer.
    return alloc_type == sycl::usm::alloc::shared || alloc_type == sycl::usm::alloc::unknown;
}

/// Throw an exception if the scalar is not accessible in the host
template <typename T>
void check_ptr_is_host_accessible(const std::string &function_name, const std::string &scalar_name,
                                  sycl::queue &queue, const T *host_or_device_ptr) {
    if (!is_ptr_accessible_on_host(queue, host_or_device_ptr)) {
        throw mkl::invalid_argument(
            "sparse_blas", function_name,
            "Scalar " + scalar_name + " must be accessible on the host for buffer functions.");
    }
}

template <typename InternalSparseMatHandleT>
void check_valid_spmm_common(const std::string function_name, sycl::queue &queue,
                             oneapi::mkl::sparse::matrix_view A_view,
                             InternalSparseMatHandleT internal_A_handle,
                             oneapi::mkl::sparse::dense_matrix_handle_t B_handle,
                             oneapi::mkl::sparse::dense_matrix_handle_t C_handle, const void *alpha,
                             const void *beta) {
    THROW_IF_NULLPTR(function_name, internal_A_handle);
    THROW_IF_NULLPTR(function_name, B_handle);
    THROW_IF_NULLPTR(function_name, C_handle);

    detail::check_all_containers_compatible(function_name, internal_A_handle, B_handle, C_handle);
    if (internal_A_handle->all_use_buffer()) {
        check_ptr_is_host_accessible("spmm", "alpha", queue, alpha);
        check_ptr_is_host_accessible("spmm", "beta", queue, beta);
    }
    if (B_handle->dense_layout != C_handle->dense_layout) {
        throw mkl::invalid_argument("sparse_blas", function_name,
                                    "B and C matrices must used the same layout.");
    }

    if (A_view.type_view != oneapi::mkl::sparse::matrix_descr::general) {
        throw mkl::invalid_argument("sparse_blas", function_name,
                                    "Matrix view's type must be `matrix_descr::general`.");
    }

    if (A_view.diag_view != oneapi::mkl::diag::nonunit) {
        throw mkl::invalid_argument("sparse_blas", function_name,
                                    "Matrix's diag_view must be `nonunit`.");
    }
}

template <typename InternalSparseMatHandleT>
void check_valid_spmv_common(const std::string function_name, sycl::queue &queue,
                             oneapi::mkl::transpose opA, oneapi::mkl::sparse::matrix_view A_view,
                             InternalSparseMatHandleT internal_A_handle,
                             oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                             oneapi::mkl::sparse::dense_vector_handle_t y_handle, const void *alpha,
                             const void *beta) {
    THROW_IF_NULLPTR(function_name, internal_A_handle);
    THROW_IF_NULLPTR(function_name, x_handle);
    THROW_IF_NULLPTR(function_name, y_handle);

    detail::check_all_containers_compatible(function_name, internal_A_handle, x_handle, y_handle);
    if (internal_A_handle->all_use_buffer()) {
        check_ptr_is_host_accessible("spmv", "alpha", queue, alpha);
        check_ptr_is_host_accessible("spmv", "beta", queue, beta);
    }
    if (A_view.type_view == oneapi::mkl::sparse::matrix_descr::diagonal) {
        throw mkl::invalid_argument("sparse_blas", function_name,
                                    "Matrix view's type cannot be diagonal.");
    }

    if (A_view.type_view != oneapi::mkl::sparse::matrix_descr::triangular &&
        A_view.diag_view == oneapi::mkl::diag::unit) {
        throw mkl::invalid_argument(
            "sparse_blas", function_name,
            "`unit` diag_view can only be used with a triangular type_view.");
    }

    if ((A_view.type_view == oneapi::mkl::sparse::matrix_descr::symmetric ||
         A_view.type_view == oneapi::mkl::sparse::matrix_descr::hermitian) &&
        opA == oneapi::mkl::transpose::conjtrans) {
        throw mkl::invalid_argument(
            "sparse_blas", function_name,
            "Symmetric or Hermitian matrix cannot be conjugated with `conjtrans`.");
    }
}

template <typename InternalSparseMatHandleT>
void check_valid_spsv_common(const std::string function_name, sycl::queue &queue,
                             oneapi::mkl::sparse::matrix_view A_view,
                             InternalSparseMatHandleT internal_A_handle,
                             oneapi::mkl::sparse::dense_vector_handle_t x_handle,
                             oneapi::mkl::sparse::dense_vector_handle_t y_handle,
                             const void *alpha) {
    THROW_IF_NULLPTR(function_name, internal_A_handle);
    THROW_IF_NULLPTR(function_name, x_handle);
    THROW_IF_NULLPTR(function_name, y_handle);

    detail::check_all_containers_compatible(function_name, internal_A_handle, x_handle, y_handle);
    if (A_view.type_view != matrix_descr::triangular) {
        throw mkl::invalid_argument("sparse_blas", function_name,
                                    "Matrix view's type must be `matrix_descr::triangular`.");
    }

    if (internal_A_handle->all_use_buffer()) {
        check_ptr_is_host_accessible("spsv", "alpha", queue, alpha);
    }
}

} // namespace oneapi::mkl::sparse::detail

#endif // _ONEMKL_SRC_SPARSE_BLAS_COMMON_OP_VERIFICATION_HPP_