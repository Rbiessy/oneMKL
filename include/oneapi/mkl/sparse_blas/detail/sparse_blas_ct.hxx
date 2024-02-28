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

// This file is meant to be included in each backend sparse_blas_ct.hpp files
// Each function calls the implementation from onemkl_sparse_blas_backends.hxx

#ifndef BACKEND
#error "BACKEND is not defined"
#endif

// Dense vector
template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> create_dense_vector(
    backend_selector<backend::BACKEND> selector, dense_vector_handle_t *p_dvhandle,
    std::int64_t size, sycl::buffer<fpType, 1> &val) {
    BACKEND::create_dense_vector(selector.get_queue(), p_dvhandle, size, val);
}
template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> create_dense_vector(
    backend_selector<backend::BACKEND> selector, dense_vector_handle_t *p_dvhandle,
    std::int64_t size, fpType *val, const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::create_dense_vector(selector.get_queue(), p_dvhandle, size, val, dependencies);
}

// Dense matrix
template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> create_dense_matrix(
    backend_selector<backend::BACKEND> selector, dense_matrix_handle_t *p_dmhandle,
    std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld, layout dense_layout,
    sycl::buffer<fpType, 1> &val) {
    BACKEND::create_dense_matrix(selector.get_queue(), p_dmhandle, num_rows, num_cols, ld,
                                 dense_layout, val);
}
template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> create_dense_matrix(
    backend_selector<backend::BACKEND> selector, dense_matrix_handle_t *p_dmhandle,
    std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld, layout dense_layout, fpType *val,
    const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::create_dense_matrix(selector.get_queue(), p_dmhandle, num_rows, num_cols, ld,
                                        dense_layout, val, dependencies);
}

// CSR matrix
template <typename fpType, typename intType>
std::enable_if_t<detail::are_fp_int_supported_v<fpType, intType>> create_csr_matrix(
    backend_selector<backend::BACKEND> selector, matrix_handle_t *p_smhandle, std::int64_t num_rows,
    std::int64_t num_cols, std::int64_t nnz, index_base index, sycl::buffer<intType, 1> &row_ptr,
    sycl::buffer<intType, 1> &col_ind, sycl::buffer<fpType, 1> &val) {
    BACKEND::create_csr_matrix(selector.get_queue(), p_smhandle, num_rows, num_cols, nnz, index,
                               row_ptr, col_ind, val);
}
template <typename fpType, typename intType>
std::enable_if_t<detail::are_fp_int_supported_v<fpType, intType>, sycl::event> create_csr_matrix(
    backend_selector<backend::BACKEND> selector, matrix_handle_t *p_smhandle, std::int64_t num_rows,
    std::int64_t num_cols, std::int64_t nnz, index_base index, intType *row_ptr, intType *col_ind,
    fpType *val, const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::create_csr_matrix(selector.get_queue(), p_smhandle, num_rows, num_cols, nnz,
                                      index, row_ptr, col_ind, val, dependencies);
}

// Destroy data types
inline sycl::event destroy_dense_vector(backend_selector<backend::BACKEND> selector,
                                        dense_vector_handle_t dvhandle,
                                        const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::destroy_dense_vector(selector.get_queue(), dvhandle, dependencies);
}
inline sycl::event destroy_dense_matrix(backend_selector<backend::BACKEND> selector,
                                        dense_matrix_handle_t dmhandle,
                                        const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::destroy_dense_matrix(selector.get_queue(), dmhandle, dependencies);
}
inline sycl::event destroy_csr_matrix(backend_selector<backend::BACKEND> selector,
                                      matrix_handle_t smhandle,
                                      const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::destroy_csr_matrix(selector.get_queue(), smhandle, dependencies);
}

// Matrix property
inline void set_matrix_property(backend_selector<backend::BACKEND> selector,
                                matrix_handle_t smhandle, matrix_property property_value) {
    BACKEND::set_matrix_property(selector.get_queue(), smhandle, property_value);
}

// Operation descriptor
template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> init_trsv_descr(
    backend_selector<backend::BACKEND> selector, trsv_descr_t *p_trsv_descr) {
    BACKEND::init_trsv_descr<fpType>(selector.get_queue(), p_trsv_descr);
}
inline sycl::event release_trsv_descr(backend_selector<backend::BACKEND> selector,
                                      trsv_descr_t trsv_descr,
                                      const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::release_trsv_descr(selector.get_queue(), trsv_descr, dependencies);
}

// Temporary buffer size
inline sycl::event trsv_buffer_size(backend_selector<backend::BACKEND> selector, uplo uplo_val,
                                    transpose transpose_val, diag diag_val,
                                    matrix_handle_t A_handle, dense_vector_handle_t x,
                                    dense_vector_handle_t y, trsv_alg alg, trsv_descr_t trsv_descr,
                                    std::int64_t &temp_buffer_size,
                                    const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::trsv_buffer_size(selector.get_queue(), uplo_val, transpose_val, diag_val,
                                     A_handle, x, y, alg, trsv_descr, temp_buffer_size,
                                     dependencies);
}

inline sycl::event optimize_gemm(backend_selector<backend::BACKEND> selector, transpose transpose_A,
                                 matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::optimize_gemm(selector.get_queue(), transpose_A, handle, dependencies);
}

inline sycl::event optimize_gemm(backend_selector<backend::BACKEND> selector, transpose transpose_A,
                                 transpose transpose_B, layout dense_matrix_layout,
                                 const std::int64_t columns, matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::optimize_gemm(selector.get_queue(), transpose_A, transpose_B,
                                  dense_matrix_layout, columns, handle, dependencies);
}

inline sycl::event optimize_gemv(backend_selector<backend::BACKEND> selector,
                                 transpose transpose_val, matrix_handle_t handle,
                                 const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::optimize_gemv(selector.get_queue(), transpose_val, handle, dependencies);
}

inline void optimize_trsv(backend_selector<backend::BACKEND> selector, uplo uplo_val,
                          transpose transpose_val, diag diag_val, matrix_handle_t A_handle,
                          trsv_alg alg, trsv_descr_t trsv_descr, std::int64_t temp_buffer_size,
                          sycl::buffer<std::uint8_t, 1> temp_buffer) {
    BACKEND::optimize_trsv(selector.get_queue(), uplo_val, transpose_val, diag_val, A_handle, alg,
                           trsv_descr, temp_buffer_size, temp_buffer);
}
inline sycl::event optimize_trsv(backend_selector<backend::BACKEND> selector, uplo uplo_val,
                                 transpose transpose_val, diag diag_val, matrix_handle_t A_handle,
                                 trsv_alg alg, trsv_descr_t trsv_descr,
                                 std::int64_t temp_buffer_size, void *temp_buffer,
                                 const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::optimize_trsv(selector.get_queue(), uplo_val, transpose_val, diag_val, A_handle,
                                  alg, trsv_descr, temp_buffer_size, temp_buffer, dependencies);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> gemv(
    backend_selector<backend::BACKEND> selector, transpose transpose_val, const fpType alpha,
    matrix_handle_t A_handle, sycl::buffer<fpType, 1> &x, const fpType beta,
    sycl::buffer<fpType, 1> &y) {
    BACKEND::gemv(selector.get_queue(), transpose_val, alpha, A_handle, x, beta, y);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> gemv(
    backend_selector<backend::BACKEND> selector, transpose transpose_val, const fpType alpha,
    matrix_handle_t A_handle, const fpType *x, const fpType beta, fpType *y,
    const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::gemv(selector.get_queue(), transpose_val, alpha, A_handle, x, beta, y,
                         dependencies);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>> gemm(
    backend_selector<backend::BACKEND> selector, layout dense_matrix_layout, transpose transpose_A,
    transpose transpose_B, const fpType alpha, matrix_handle_t A_handle, sycl::buffer<fpType, 1> &B,
    const std::int64_t columns, const std::int64_t ldb, const fpType beta,
    sycl::buffer<fpType, 1> &C, const std::int64_t ldc) {
    BACKEND::gemm(selector.get_queue(), dense_matrix_layout, transpose_A, transpose_B, alpha,
                  A_handle, B, columns, ldb, beta, C, ldc);
}

template <typename fpType>
std::enable_if_t<detail::is_fp_supported_v<fpType>, sycl::event> gemm(
    backend_selector<backend::BACKEND> selector, layout dense_matrix_layout, transpose transpose_A,
    transpose transpose_B, const fpType alpha, matrix_handle_t A_handle, const fpType *B,
    const std::int64_t columns, const std::int64_t ldb, const fpType beta, fpType *C,
    const std::int64_t ldc, const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::gemm(selector.get_queue(), dense_matrix_layout, transpose_A, transpose_B, alpha,
                         A_handle, B, columns, ldb, beta, C, ldc, dependencies);
}

inline sycl::event trsv(backend_selector<backend::BACKEND> selector, uplo uplo_val,
                        transpose transpose_val, diag diag_val, matrix_handle_t A_handle,
                        dense_vector_handle_t x, dense_vector_handle_t y, trsv_alg alg,
                        trsv_descr_t trsv_descr,
                        const std::vector<sycl::event> &dependencies = {}) {
    return BACKEND::trsv(selector.get_queue(), uplo_val, transpose_val, diag_val, A_handle, x, y,
                         alg, trsv_descr, dependencies);
}
