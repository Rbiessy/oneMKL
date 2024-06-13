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

#ifndef _ONEMKL_SRC_SPARSE_BLAS_BACKENDS_CUSPARSE_HANDLES_HPP_
#define _ONEMKL_SRC_SPARSE_BLAS_BACKENDS_CUSPARSE_HANDLES_HPP_

#include <cusparse.h>

#include "sparse_blas/generic_container.hpp"

namespace oneapi::mkl::sparse {

// Complete the definition of incomplete types dense_vector_handle, dense_matrix_handle and matrix_handle.

struct dense_vector_handle : public detail::generic_dense_vector_handle<cusparseDnVecDescr_t> {
    template <typename T>
    dense_vector_handle(cusparseDnVecDescr_t cu_descr, T* value_ptr, std::int64_t size)
            : detail::generic_dense_vector_handle<cusparseDnVecDescr_t>(cu_descr, value_ptr, size) {
    }

    template <typename T>
    dense_vector_handle(cusparseDnVecDescr_t cu_descr, const sycl::buffer<T, 1> value_buffer,
                        std::int64_t size)
            : detail::generic_dense_vector_handle<cusparseDnVecDescr_t>(cu_descr, value_buffer,
                                                                        size) {}
};

struct dense_matrix_handle : public detail::generic_dense_matrix_handle<cusparseDnMatDescr_t> {
    template <typename T>
    dense_matrix_handle(cusparseDnMatDescr_t cu_descr, T* value_ptr, std::int64_t num_rows,
                        std::int64_t num_cols, std::int64_t ld, layout dense_layout)
            : detail::generic_dense_matrix_handle<cusparseDnMatDescr_t>(
                  cu_descr, value_ptr, num_rows, num_cols, ld, dense_layout) {}

    template <typename T>
    dense_matrix_handle(cusparseDnMatDescr_t cu_descr, const sycl::buffer<T, 1> value_buffer,
                        std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,
                        layout dense_layout)
            : detail::generic_dense_matrix_handle<cusparseDnMatDescr_t>(
                  cu_descr, value_buffer, num_rows, num_cols, ld, dense_layout) {}
};

struct matrix_handle : public detail::generic_sparse_handle<cusparseSpMatDescr_t> {
    template <typename fpType, typename intType>
    matrix_handle(cusparseSpMatDescr_t cu_descr, intType* row_ptr, intType* col_ptr,
                  fpType* value_ptr, std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                  oneapi::mkl::index_base index)
            : detail::generic_sparse_handle<cusparseSpMatDescr_t>(
                  cu_descr, row_ptr, col_ptr, value_ptr, num_rows, num_cols, nnz, index) {}

    template <typename fpType, typename intType>
    matrix_handle(cusparseSpMatDescr_t cu_descr, const sycl::buffer<intType, 1> row_buffer,
                  const sycl::buffer<intType, 1> col_buffer,
                  const sycl::buffer<fpType, 1> value_buffer, std::int64_t num_rows,
                  std::int64_t num_cols, std::int64_t nnz, oneapi::mkl::index_base index)
            : detail::generic_sparse_handle<cusparseSpMatDescr_t>(
                  cu_descr, row_buffer, col_buffer, value_buffer, num_rows, num_cols, nnz, index) {}
};

} // namespace oneapi::mkl::sparse

namespace oneapi::mkl::sparse::detail {

/**
 * Internal matrix_handle type for MKL backends.
 * Here \p matrix_handle_t is the type of the backend's handle.
 * The user-facing incomplete type matrix_handle_t must be kept incomplete.
 * Internally matrix_handle_t is reinterpret_cast as oneapi::mkl::sparse::detail::matrix_handle which holds another matrix_handle_t for the backend handle.
 */
using matrix_handle = detail::generic_sparse_handle<matrix_handle_t>;

/// Cast to oneMKL's interface handle type
inline auto get_internal_handle(matrix_handle_t handle) {
    return reinterpret_cast<matrix_handle*>(handle);
}

} // namespace oneapi::mkl::sparse::detail

#endif // _ONEMKL_SRC_SPARSE_BLAS_BACKENDS_CUSPARSE_HANDLES_HPP_
