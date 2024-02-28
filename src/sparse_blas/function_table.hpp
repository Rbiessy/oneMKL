/*******************************************************************************
* Copyright 2023 Codeplay Software Ltd.
*
* (*Licensed under the Apache License, Version 2.0 )(the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#ifndef _ONEMKL_SPARSE_BLAS_FUNCTION_TABLE_HPP_
#define _ONEMKL_SPARSE_BLAS_FUNCTION_TABLE_HPP_

#include "oneapi/mkl/sparse_blas/types.hpp"
#include "sparse_blas/macros.hpp"

#define DEFINE_CREATE_DENSE_VECTOR(FP_TYPE, FP_SUFFIX)                                \
    void (*create_dense_vector_buffer##FP_SUFFIX)(                                    \
        sycl::queue & queue, oneapi::mkl::sparse::dense_vector_handle_t * p_dvhandle, \
        std::int64_t size, sycl::buffer<FP_TYPE, 1> & val);                           \
    sycl::event (*create_dense_vector_usm##FP_SUFFIX)(                                \
        sycl::queue & queue, oneapi::mkl::sparse::dense_vector_handle_t * p_dvhandle, \
        std::int64_t size, FP_TYPE * val, const std::vector<sycl::event> &dependencies)

#define DEFINE_CREATE_DENSE_MATRIX(FP_TYPE, FP_SUFFIX)                                \
    void (*create_dense_matrix_buffer##FP_SUFFIX)(                                    \
        sycl::queue & queue, oneapi::mkl::sparse::dense_matrix_handle_t * p_dmhandle, \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,                \
        oneapi::mkl::layout dense_layout, sycl::buffer<FP_TYPE, 1> & val);            \
    sycl::event (*create_dense_matrix_usm##FP_SUFFIX)(                                \
        sycl::queue & queue, oneapi::mkl::sparse::dense_matrix_handle_t * p_dmhandle, \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,                \
        oneapi::mkl::layout dense_layout, FP_TYPE * val,                              \
        const std::vector<sycl::event> &dependencies)

#define DEFINE_CREATE_CSR_MATRIX(FP_TYPE, FP_SUFFIX, INT_TYPE, INT_SUFFIX)                    \
    void (*create_csr_matrix_buffer##FP_SUFFIX##INT_SUFFIX)(                                  \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t * p_smhandle,               \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                       \
        oneapi::mkl::index_base index, sycl::buffer<INT_TYPE, 1> & row_ptr,                   \
        sycl::buffer<INT_TYPE, 1> & col_ind, sycl::buffer<FP_TYPE, 1> & val);                 \
    sycl::event (*create_csr_matrix_usm##FP_SUFFIX##INT_SUFFIX)(                              \
        sycl::queue & queue, oneapi::mkl::sparse::matrix_handle_t * p_smhandle,               \
        std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,                       \
        oneapi::mkl::index_base index, INT_TYPE * row_ptr, INT_TYPE * col_ind, FP_TYPE * val, \
        const std::vector<sycl::event> &dependencies)

#define DEFINE_INIT_TRSV_DESCR(FP_TYPE, FP_SUFFIX)          \
    void (*init_trsv_descr##FP_SUFFIX)(sycl::queue & queue, \
                                       oneapi::mkl::sparse::trsv_descr_t * p_trsv_descr)

#define DEFINE_GEMV(FP_TYPE, FP_SUFFIX)                                                      \
    void (*gemv_buffer##FP_SUFFIX)(                                                          \
        sycl::queue & queue, oneapi::mkl::transpose transpose_val, const FP_TYPE alpha,      \
        oneapi::mkl::sparse::matrix_handle_t A_handle, sycl::buffer<FP_TYPE, 1> &x,          \
        const FP_TYPE beta, sycl::buffer<FP_TYPE, 1> &y);                                    \
    sycl::event (*gemv_usm##FP_SUFFIX)(                                                      \
        sycl::queue & queue, oneapi::mkl::transpose transpose_val, const FP_TYPE alpha,      \
        oneapi::mkl::sparse::matrix_handle_t A_handle, const FP_TYPE *x, const FP_TYPE beta, \
        FP_TYPE *y, const std::vector<sycl::event> &dependencies)

#define DEFINE_GEMM(FP_TYPE, FP_SUFFIX)                                                       \
    void (*gemm_buffer##FP_SUFFIX)(                                                           \
        sycl::queue & queue, oneapi::mkl::layout dense_matrix_layout,                         \
        oneapi::mkl::transpose transpose_A, oneapi::mkl::transpose transpose_B,               \
        const FP_TYPE alpha, oneapi::mkl::sparse::matrix_handle_t A_handle,                   \
        sycl::buffer<FP_TYPE, 1> &B, const std::int64_t columns, const std::int64_t ldb,      \
        const FP_TYPE beta, sycl::buffer<FP_TYPE, 1> &C, const std::int64_t ldc);             \
    sycl::event (*gemm_usm##FP_SUFFIX)(                                                       \
        sycl::queue & queue, oneapi::mkl::layout dense_matrix_layout,                         \
        oneapi::mkl::transpose transpose_A, oneapi::mkl::transpose transpose_B,               \
        const FP_TYPE alpha, oneapi::mkl::sparse::matrix_handle_t A_handle, const FP_TYPE *B, \
        const std::int64_t columns, const std::int64_t ldb, const FP_TYPE beta, FP_TYPE *C,   \
        const std::int64_t ldc, const std::vector<sycl::event> &dependencies)

typedef struct {
    int version;

    FOR_EACH_FP_TYPE(DEFINE_CREATE_DENSE_VECTOR);
    FOR_EACH_FP_TYPE(DEFINE_CREATE_DENSE_MATRIX);
    FOR_EACH_FP_AND_INT_TYPE(DEFINE_CREATE_CSR_MATRIX);

    // Destroy data types
    sycl::event (*destroy_dense_vector)(sycl::queue &queue,
                                        oneapi::mkl::sparse::dense_vector_handle_t dvhandle,
                                        const std::vector<sycl::event> &dependencies);
    sycl::event (*destroy_dense_matrix)(sycl::queue &queue,
                                        oneapi::mkl::sparse::dense_matrix_handle_t dmhandle,
                                        const std::vector<sycl::event> &dependencies);
    sycl::event (*destroy_csr_matrix)(sycl::queue &queue,
                                      oneapi::mkl::sparse::matrix_handle_t smhandle,
                                      const std::vector<sycl::event> &dependencies);

    // Matrix property
    void (*set_matrix_property)(sycl::queue &queue, oneapi::mkl::sparse::matrix_handle_t smhandle,
                                oneapi::mkl::sparse::matrix_property property_value);

    // Operation descriptor
    FOR_EACH_FP_TYPE(DEFINE_INIT_TRSV_DESCR);
    sycl::event (*release_trsv_descr)(sycl::queue &queue,
                                      oneapi::mkl::sparse::trsv_descr_t trsv_descr,
                                      const std::vector<sycl::event> &dependencies);

    // Temporary buffer size
    sycl::event (*trsv_buffer_size)(
        sycl::queue &queue, oneapi::mkl::uplo uplo_val, oneapi::mkl::transpose transpose_val,
        oneapi::mkl::diag diag_val, oneapi::mkl::sparse::matrix_handle_t A_handle,
        oneapi::mkl::sparse::dense_vector_handle_t x, oneapi::mkl::sparse::dense_vector_handle_t y,
        oneapi::mkl::sparse::trsv_alg alg, oneapi::mkl::sparse::trsv_descr_t trsv_descr,
        std::int64_t &temp_buffer_size, const std::vector<sycl::event> &dependencies);

    // optimize_*
    sycl::event (*optimize_gemm_v1)(sycl::queue &queue, oneapi::mkl::transpose transpose_A,
                                    oneapi::mkl::sparse::matrix_handle_t handle,
                                    const std::vector<sycl::event> &dependencies);
    sycl::event (*optimize_gemm_v2)(sycl::queue &queue, oneapi::mkl::transpose transpose_A,
                                    oneapi::mkl::transpose transpose_B,
                                    oneapi::mkl::layout dense_matrix_layout,
                                    const std::int64_t columns,
                                    oneapi::mkl::sparse::matrix_handle_t handle,
                                    const std::vector<sycl::event> &dependencies);
    sycl::event (*optimize_gemv)(sycl::queue &queue, oneapi::mkl::transpose transpose_val,
                                 oneapi::mkl::sparse::matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies);
    void (*optimize_trsv_buffer)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                                 oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                                 oneapi::mkl::sparse::matrix_handle_t A_handle,
                                 oneapi::mkl::sparse::trsv_alg alg,
                                 oneapi::mkl::sparse::trsv_descr_t trsv_descr,
                                 std::int64_t temp_buffer_size,
                                 sycl::buffer<std::uint8_t, 1> temp_buffer);
    sycl::event (*optimize_trsv_usm)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                                     oneapi::mkl::transpose transpose_val,
                                     oneapi::mkl::diag diag_val,
                                     oneapi::mkl::sparse::matrix_handle_t A_handle,
                                     oneapi::mkl::sparse::trsv_alg alg,
                                     oneapi::mkl::sparse::trsv_descr_t trsv_descr,
                                     std::int64_t temp_buffer_size, void *temp_buffer,
                                     const std::vector<sycl::event> &dependencies);

    FOR_EACH_FP_TYPE(DEFINE_GEMV);
    FOR_EACH_FP_TYPE(DEFINE_GEMM);

    sycl::event (*trsv)(sycl::queue &queue, oneapi::mkl::uplo uplo_val,
                        oneapi::mkl::transpose transpose_val, oneapi::mkl::diag diag_val,
                        oneapi::mkl::sparse::matrix_handle_t A_handle,
                        oneapi::mkl::sparse::dense_vector_handle_t x,
                        oneapi::mkl::sparse::dense_vector_handle_t y,
                        oneapi::mkl::sparse::trsv_alg alg,
                        oneapi::mkl::sparse::trsv_descr_t trsv_descr,
                        const std::vector<sycl::event> &dependencies);

} sparse_blas_function_table_t;

#undef DEFINE_CREATE_DENSE_VECTOR
#undef DEFINE_CREATE_DENSE_MATRIX
#undef DEFINE_CREATE_CSR_MATRIX
#undef DEFINE_INIT_TRSV_DESCR
#undef DEFINE_GEMV
#undef DEFINE_GEMM

#endif // _ONEMKL_SPARSE_BLAS_FUNCTION_TABLE_HPP_
