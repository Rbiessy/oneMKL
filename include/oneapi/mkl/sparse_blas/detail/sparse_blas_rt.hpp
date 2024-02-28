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

#ifndef _ONEMKL_SPARSE_BLAS_DETAIL_SPARSE_BLAS_RT_HPP_
#define _ONEMKL_SPARSE_BLAS_DETAIL_SPARSE_BLAS_RT_HPP_

#include "oneapi/mkl/sparse_blas/detail/helper_types.hpp"
#include "oneapi/mkl/sparse_blas/types.hpp"

namespace oneapi {
namespace mkl {
namespace sparse {

// TODO(Romain): Make functions create_*, destroy_* and *_buffer_size synchronous
// TODO(Romain): Update API for gemm and gemv operations
// TODO(Romain): Introduce matrix_view type to store some "matrix description" enum, uplo_val and diag_val

// Dense vector
template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> create_dense_vector(
    sycl::queue &queue, dense_vector_handle_t *p_dvhandle, std::int64_t size,
    sycl::buffer<fpType, 1> &val);
template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> create_dense_vector(
    sycl::queue &queue, dense_vector_handle_t *p_dvhandle, std::int64_t size, fpType *val,
    const std::vector<sycl::event> &dependencies = {});

// Dense matrix
template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> create_dense_matrix(
    sycl::queue &queue, dense_matrix_handle_t *p_dmhandle, std::int64_t num_rows,
    std::int64_t num_cols, std::int64_t ld, layout dense_layout, sycl::buffer<fpType, 1> &val);
template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> create_dense_matrix(
    sycl::queue &queue, dense_matrix_handle_t *p_dmhandle, std::int64_t num_rows,
    std::int64_t num_cols, std::int64_t ld, layout dense_layout, fpType *val,
    const std::vector<sycl::event> &dependencies = {});

// CSR matrix
template <typename fpType, typename intType>
std::enable_if_t<detail::are_fp_int_supported_v<fpType, intType>> create_csr_matrix(
    sycl::queue &queue, matrix_handle_t *p_smhandle, std::int64_t num_rows, std::int64_t num_cols,
    std::int64_t nnz, index_base index, sycl::buffer<intType, 1> &row_ptr,
    sycl::buffer<intType, 1> &col_ind, sycl::buffer<fpType, 1> &val);
template <typename fpType, typename intType>
std::enable_if_t<detail::are_fp_int_supported_v<fpType, intType>, sycl::event> create_csr_matrix(
    sycl::queue &queue, matrix_handle_t *p_smhandle, std::int64_t num_rows, std::int64_t num_cols,
    std::int64_t nnz, index_base index, intType *row_ptr, intType *col_ind, fpType *val,
    const std::vector<sycl::event> &dependencies = {});

// Destroy data types
sycl::event destroy_dense_vector(sycl::queue &queue, dense_vector_handle_t dvhandle,
                                 const std::vector<sycl::event> &dependencies = {});
sycl::event destroy_dense_matrix(sycl::queue &queue, dense_matrix_handle_t dmhandle,
                                 const std::vector<sycl::event> &dependencies = {});
sycl::event destroy_csr_matrix(sycl::queue &queue, matrix_handle_t smhandle,
                               const std::vector<sycl::event> &dependencies = {});

// Matrix property
// TODO(Romain): Provide another overload that throw if a property is unsupported?
void set_matrix_property(sycl::queue &queue, matrix_handle_t smhandle,
                         matrix_property property_value);

// TODO(Romain): Add support for setting matrices and vector data

// Operation descriptor
// TODO(Romain): Remove template fpType, not needed after all
template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> init_trsv_descr(sycl::queue &queue,
                                                                    trsv_descr_t *p_trsv_descr);
// TODO(Romain): Make synchronous
sycl::event release_trsv_descr(sycl::queue &queue, trsv_descr_t trsv_descr,
                               const std::vector<sycl::event> &dependencies = {});

// Temporary buffer size
// TODO(Romain): Make synchronous
sycl::event trsv_buffer_size(sycl::queue &queue, uplo uplo_val, transpose transpose_val,
                             diag diag_val, matrix_handle_t A_handle, dense_vector_handle_t x,
                             dense_vector_handle_t y, trsv_alg alg, trsv_descr_t trsv_descr,
                             std::int64_t &temp_buffer_size,
                             const std::vector<sycl::event> &dependencies = {});

// Optimize
sycl::event optimize_gemm(sycl::queue &queue, transpose transpose_A, matrix_handle_t handle,
                          const std::vector<sycl::event> &dependencies = {});

sycl::event optimize_gemm(sycl::queue &queue, transpose transpose_A, transpose transpose_B,
                          layout dense_matrix_layout, const std::int64_t columns,
                          matrix_handle_t handle,
                          const std::vector<sycl::event> &dependencies = {});

sycl::event optimize_gemv(sycl::queue &queue, transpose transpose_val, matrix_handle_t handle,
                          const std::vector<sycl::event> &dependencies = {});

// TODO(Romain): Make temp_buffer_size unsigned
void optimize_trsv(sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,
                   matrix_handle_t A_handle, trsv_alg alg, trsv_descr_t trsv_descr,
                   std::int64_t temp_buffer_size, sycl::buffer<std::uint8_t, 1> temp_buffer);
// TODO(Romain): Make synchronous?
// TODO(Romain): For trsv only, optimize_trsv should be renamed analyse_trsv and be mandatory. Other optimize_* functions are optional.
sycl::event optimize_trsv(sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,
                          matrix_handle_t A_handle, trsv_alg alg, trsv_descr_t trsv_descr,
                          std::int64_t temp_buffer_size, void *temp_buffer,
                          const std::vector<sycl::event> &dependencies = {});

// Operations
template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> gemv(
    sycl::queue &queue, transpose transpose_val, const fpType alpha, matrix_handle_t A_handle,
    sycl::buffer<fpType, 1> &x, const fpType beta, sycl::buffer<fpType, 1> &y);

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> gemv(
    sycl::queue &queue, transpose transpose_val, const fpType alpha, matrix_handle_t A_handle,
    const fpType *x, const fpType beta, fpType *y,
    const std::vector<sycl::event> &dependencies = {});

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> gemm(
    sycl::queue &queue, layout dense_matrix_layout, transpose transpose_A, transpose transpose_B,
    const fpType alpha, matrix_handle_t A_handle, sycl::buffer<fpType, 1> &B,
    const std::int64_t columns, const std::int64_t ldb, const fpType beta,
    sycl::buffer<fpType, 1> &C, const std::int64_t ldc);

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> gemm(
    sycl::queue &queue, layout dense_matrix_layout, transpose transpose_A, transpose transpose_B,
    const fpType alpha, matrix_handle_t A_handle, const fpType *B, const std::int64_t columns,
    const std::int64_t ldb, const fpType beta, fpType *C, const std::int64_t ldc,
    const std::vector<sycl::event> &dependencies = {});

sycl::event trsv(sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,
                 matrix_handle_t A_handle, dense_vector_handle_t x, dense_vector_handle_t y,
                 trsv_alg alg, trsv_descr_t trsv_descr,
                 const std::vector<sycl::event> &dependencies = {});

} // namespace sparse
} // namespace mkl
} // namespace oneapi

#endif // _ONEMKL_SPARSE_BLAS_DETAIL_SPARSE_BLAS_RT_HPP_
