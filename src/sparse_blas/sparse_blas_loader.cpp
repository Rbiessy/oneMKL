/*******************************************************************************
* Copyright 2023 Codeplay Software Ltd.
*
* Licensed under the Apache License, Version 2.0 (the "License");
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

#include "oneapi/mkl/sparse_blas/detail/sparse_blas_rt.hpp"

#include "function_table_initializer.hpp"
#include "sparse_blas/function_table.hpp"
#include "sparse_blas/macros.hpp"
#include "oneapi/mkl/detail/get_device_id.hpp"

namespace oneapi::mkl::sparse {

static oneapi::mkl::detail::table_initializer<mkl::domain::sparse_blas,
                                              sparse_blas_function_table_t>
    function_tables;

#define DEFINE_CREATE_DENSE_VECTOR(FP_TYPE, FP_SUFFIX)                                             \
    template <>                                                                                    \
    void create_dense_vector(sycl::queue &queue, dense_vector_handle_t *p_dvhandle,                \
                             std::int64_t size, sycl::buffer<FP_TYPE, 1> &val) {                   \
        auto libkey = get_device_id(queue);                                                        \
        function_tables[libkey].create_dense_vector_buffer##FP_SUFFIX(queue, p_dvhandle, size,     \
                                                                      val);                        \
    }                                                                                              \
    template <>                                                                                    \
    sycl::event create_dense_vector(sycl::queue &queue, dense_vector_handle_t *p_dvhandle,         \
                                    std::int64_t size, FP_TYPE *val,                               \
                                    const std::vector<sycl::event> &dependencies) {                \
        auto libkey = get_device_id(queue);                                                        \
        return function_tables[libkey].create_dense_vector_usm##FP_SUFFIX(queue, p_dvhandle, size, \
                                                                          val, dependencies);      \
    }
FOR_EACH_FP_TYPE(DEFINE_CREATE_DENSE_VECTOR)
#undef DEFINE_CREATE_DENSE_VECTOR

#define DEFINE_CREATE_DENSE_MATRIX(FP_TYPE, FP_SUFFIX)                                             \
    template <>                                                                                    \
    void create_dense_matrix(sycl::queue &queue, dense_matrix_handle_t *p_dmhandle,                \
                             std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,        \
                             layout dense_layout, sycl::buffer<FP_TYPE, 1> &val) {                 \
        auto libkey = get_device_id(queue);                                                        \
        function_tables[libkey].create_dense_matrix_buffer##FP_SUFFIX(                             \
            queue, p_dmhandle, num_rows, num_cols, ld, dense_layout, val);                         \
    }                                                                                              \
    template <>                                                                                    \
    sycl::event create_dense_matrix(sycl::queue &queue, dense_matrix_handle_t *p_dmhandle,         \
                                    std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld, \
                                    layout dense_layout, FP_TYPE *val,                             \
                                    const std::vector<sycl::event> &dependencies) {                \
        auto libkey = get_device_id(queue);                                                        \
        return function_tables[libkey].create_dense_matrix_usm##FP_SUFFIX(                         \
            queue, p_dmhandle, num_rows, num_cols, ld, dense_layout, val, dependencies);           \
    }
FOR_EACH_FP_TYPE(DEFINE_CREATE_DENSE_MATRIX)
#undef DEFINE_CREATE_DENSE_MATRIX

#define DEFINE_CREATE_CSR_MATRIX(FP_TYPE, FP_SUFFIX, INT_TYPE, INT_SUFFIX)                         \
    template <>                                                                                    \
    void create_csr_matrix(sycl::queue &queue, matrix_handle_t *p_smhandle, std::int64_t num_rows, \
                           std::int64_t num_cols, std::int64_t nnz, index_base index,              \
                           sycl::buffer<INT_TYPE, 1> &row_ptr, sycl::buffer<INT_TYPE, 1> &col_ind, \
                           sycl::buffer<FP_TYPE, 1> &val) {                                        \
        auto libkey = get_device_id(queue);                                                        \
        function_tables[libkey].create_csr_matrix_buffer##FP_SUFFIX##INT_SUFFIX(                   \
            queue, p_smhandle, num_rows, num_cols, nnz, index, row_ptr, col_ind, val);             \
    }                                                                                              \
    template <>                                                                                    \
    sycl::event create_csr_matrix(sycl::queue &queue, matrix_handle_t *p_smhandle,                 \
                                  std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,  \
                                  index_base index, INT_TYPE *row_ptr, INT_TYPE *col_ind,          \
                                  FP_TYPE *val, const std::vector<sycl::event> &dependencies) {    \
        auto libkey = get_device_id(queue);                                                        \
        return function_tables[libkey].create_csr_matrix_usm##FP_SUFFIX##INT_SUFFIX(               \
            queue, p_smhandle, num_rows, num_cols, nnz, index, row_ptr, col_ind, val,              \
            dependencies);                                                                         \
    }

FOR_EACH_FP_AND_INT_TYPE(DEFINE_CREATE_CSR_MATRIX)
#undef DEFINE_CREATE_CSR_MATRIX

sycl::event destroy_dense_vector(sycl::queue &queue, dense_vector_handle_t p_dvhandle,
                                 const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].destroy_dense_vector(queue, p_dvhandle, dependencies);
}

sycl::event destroy_dense_matrix(sycl::queue &queue, dense_matrix_handle_t p_dmhandle,
                                 const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].destroy_dense_matrix(queue, p_dmhandle, dependencies);
}

sycl::event destroy_csr_matrix(sycl::queue &queue, matrix_handle_t p_smhandle,
                               const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].destroy_csr_matrix(queue, p_smhandle, dependencies);
}

void set_matrix_property(sycl::queue &queue, matrix_handle_t smhandle,
                         matrix_property property_value) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].set_matrix_property(queue, smhandle, property_value);
}

#define DEFINE_INIT_TRSV_DESCR(FP_TYPE, FP_SUFFIX)                                      \
    template <>                                                                         \
    void init_trsv_descr<FP_TYPE>(sycl::queue & queue, trsv_descr_t * p_trsv_descr) {   \
        auto libkey = get_device_id(queue);                                             \
        return function_tables[libkey].init_trsv_descr##FP_SUFFIX(queue, p_trsv_descr); \
    }
FOR_EACH_FP_TYPE(DEFINE_INIT_TRSV_DESCR)
#undef DEFINE_INIT_TRSV_DESCR

sycl::event release_trsv_descr(sycl::queue &queue, trsv_descr_t trsv_descr,
                               const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].release_trsv_descr(queue, trsv_descr, dependencies);
}

// Temporary buffer size
sycl::event trsv_buffer_size(sycl::queue &queue, uplo uplo_val, transpose transpose_val,
                             diag diag_val, matrix_handle_t A_handle, dense_vector_handle_t x,
                             dense_vector_handle_t y, trsv_alg alg, trsv_descr_t trsv_descr,
                             std::int64_t &temp_buffer_size,
                             const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].trsv_buffer_size(queue, uplo_val, transpose_val, diag_val,
                                                    A_handle, x, y, alg, trsv_descr,
                                                    temp_buffer_size, dependencies);
}

sycl::event optimize_gemm(sycl::queue &queue, transpose transpose_A, matrix_handle_t handle,
                          const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].optimize_gemm_v1(queue, transpose_A, handle, dependencies);
}

sycl::event optimize_gemm(sycl::queue &queue, transpose transpose_A, transpose transpose_B,
                          layout dense_matrix_layout, const std::int64_t columns,
                          matrix_handle_t handle, const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].optimize_gemm_v2(
        queue, transpose_A, transpose_B, dense_matrix_layout, columns, handle, dependencies);
}

sycl::event optimize_gemv(sycl::queue &queue, transpose transpose_val, matrix_handle_t handle,
                          const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].optimize_gemv(queue, transpose_val, handle, dependencies);
}

void optimize_trsv(sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,
                   matrix_handle_t A_handle, trsv_alg alg, trsv_descr_t trsv_descr,
                   std::int64_t temp_buffer_size, sycl::buffer<std::uint8_t, 1> temp_buffer) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].optimize_trsv_buffer(queue, uplo_val, transpose_val, diag_val,
                                                        A_handle, alg, trsv_descr, temp_buffer_size,
                                                        temp_buffer);
}

sycl::event optimize_trsv(sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,
                          matrix_handle_t A_handle, trsv_alg alg, trsv_descr_t trsv_descr,
                          std::int64_t temp_buffer_size, void *temp_buffer,
                          const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].optimize_trsv_usm(queue, uplo_val, transpose_val, diag_val,
                                                     A_handle, alg, trsv_descr, temp_buffer_size,
                                                     temp_buffer, dependencies);
}

#define DEFINE_GEMV(FP_TYPE, FP_SUFFIX)                                                           \
    template <>                                                                                   \
    void gemv(sycl::queue &queue, transpose transpose_val, const FP_TYPE alpha,                   \
              matrix_handle_t A_handle, sycl::buffer<FP_TYPE, 1> &x, const FP_TYPE beta,          \
              sycl::buffer<FP_TYPE, 1> &y) {                                                      \
        auto libkey = get_device_id(queue);                                                       \
        function_tables[libkey].gemv_buffer##FP_SUFFIX(queue, transpose_val, alpha, A_handle, x,  \
                                                       beta, y);                                  \
    }                                                                                             \
    template <>                                                                                   \
    sycl::event gemv(sycl::queue &queue, transpose transpose_val, const FP_TYPE alpha,            \
                     matrix_handle_t A_handle, const FP_TYPE *x, const FP_TYPE beta, FP_TYPE *y,  \
                     const std::vector<sycl::event> &dependencies) {                              \
        auto libkey = get_device_id(queue);                                                       \
        return function_tables[libkey].gemv_usm##FP_SUFFIX(queue, transpose_val, alpha, A_handle, \
                                                           x, beta, y, dependencies);             \
    }

FOR_EACH_FP_TYPE(DEFINE_GEMV)
#undef DEFINE_GEMV

#define DEFINE_GEMM(FP_TYPE, FP_SUFFIX)                                                          \
    template <>                                                                                  \
    void gemm(sycl::queue &queue, layout dense_matrix_layout, transpose transpose_A,             \
              transpose transpose_B, const FP_TYPE alpha, matrix_handle_t A_handle,              \
              sycl::buffer<FP_TYPE, 1> &B, const std::int64_t columns, const std::int64_t ldb,   \
              const FP_TYPE beta, sycl::buffer<FP_TYPE, 1> &C, const std::int64_t ldc) {         \
        auto libkey = get_device_id(queue);                                                      \
        function_tables[libkey].gemm_buffer##FP_SUFFIX(queue, dense_matrix_layout, transpose_A,  \
                                                       transpose_B, alpha, A_handle, B, columns, \
                                                       ldb, beta, C, ldc);                       \
    }                                                                                            \
    template <>                                                                                  \
    sycl::event gemm(sycl::queue &queue, layout dense_matrix_layout, transpose transpose_A,      \
                     transpose transpose_B, const FP_TYPE alpha, matrix_handle_t A_handle,       \
                     const FP_TYPE *B, const std::int64_t columns, const std::int64_t ldb,       \
                     const FP_TYPE beta, FP_TYPE *C, const std::int64_t ldc,                     \
                     const std::vector<sycl::event> &dependencies) {                             \
        auto libkey = get_device_id(queue);                                                      \
        return function_tables[libkey].gemm_usm##FP_SUFFIX(                                      \
            queue, dense_matrix_layout, transpose_A, transpose_B, alpha, A_handle, B, columns,   \
            ldb, beta, C, ldc, dependencies);                                                    \
    }

FOR_EACH_FP_TYPE(DEFINE_GEMM)
#undef DEFINE_GEMM

sycl::event trsv(sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,
                 matrix_handle_t A_handle, dense_vector_handle_t x, dense_vector_handle_t y,
                 trsv_alg alg, trsv_descr_t trsv_descr,
                 const std::vector<sycl::event> &dependencies) {
    auto libkey = get_device_id(queue);
    return function_tables[libkey].trsv(queue, uplo_val, transpose_val, diag_val, A_handle, x, y,
                                        alg, trsv_descr, dependencies);
}

} // namespace oneapi::mkl::sparse
