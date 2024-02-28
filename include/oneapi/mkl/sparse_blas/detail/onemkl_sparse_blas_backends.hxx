/***************************************************************************
*  Copyright(C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0(the "License");
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

// This file is meant to be included in each backend onemkl_sparse_blas_BACKEND.hpp files.
// It is used to exports each symbol to the onemkl_sparse_blas_BACKEND library.

// Dense vector
template <typename fpType>
ONEMKL_EXPORT void create_dense_vector(sycl::queue &queue, dense_vector_handle_t *p_dvhandle,
                                       std::int64_t size, sycl::buffer<fpType, 1> &val);
template <typename fpType>
ONEMKL_EXPORT sycl::event create_dense_vector(sycl::queue &queue, dense_vector_handle_t *p_dvhandle,
                                              std::int64_t size, fpType *val,
                                              const std::vector<sycl::event> &dependencies = {});

// Dense matrix
template <typename fpType>
ONEMKL_EXPORT void create_dense_matrix(sycl::queue &queue, dense_matrix_handle_t *p_dmhandle,
                                       std::int64_t num_rows, std::int64_t num_cols,
                                       std::int64_t ld, layout dense_layout,
                                       sycl::buffer<fpType, 1> &val);
template <typename fpType>
ONEMKL_EXPORT sycl::event create_dense_matrix(sycl::queue &queue, dense_matrix_handle_t *p_dmhandle,
                                              std::int64_t num_rows, std::int64_t num_cols,
                                              std::int64_t ld, layout dense_layout, fpType *val,
                                              const std::vector<sycl::event> &dependencies = {});

// CSR matrix
template <typename fpType, typename intType>
ONEMKL_EXPORT void create_csr_matrix(sycl::queue &queue, matrix_handle_t *p_smhandle,
                                     std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                                     index_base index, sycl::buffer<intType, 1> &row_ptr,
                                     sycl::buffer<intType, 1> &col_ind,
                                     sycl::buffer<fpType, 1> &val);
template <typename fpType, typename intType>
ONEMKL_EXPORT sycl::event create_csr_matrix(sycl::queue &queue, matrix_handle_t *p_smhandle,
                                            std::int64_t num_rows, std::int64_t num_cols,
                                            std::int64_t nnz, index_base index, intType *row_ptr,
                                            intType *col_ind, fpType *val,
                                            const std::vector<sycl::event> &dependencies = {});

// Destroy data types
ONEMKL_EXPORT sycl::event destroy_dense_vector(sycl::queue &queue, dense_vector_handle_t dvhandle,
                                               const std::vector<sycl::event> &dependencies = {});
ONEMKL_EXPORT sycl::event destroy_dense_matrix(sycl::queue &queue, dense_matrix_handle_t dmhandle,
                                               const std::vector<sycl::event> &dependencies = {});
ONEMKL_EXPORT sycl::event destroy_csr_matrix(sycl::queue &queue, matrix_handle_t smhandle,
                                             const std::vector<sycl::event> &dependencies = {});

// Matrix property
ONEMKL_EXPORT void set_matrix_property(sycl::queue &queue, matrix_handle_t smhandle,
                                       matrix_property property_value);

// Operation descriptor
template <typename fpType>
ONEMKL_EXPORT void init_trsv_descr(sycl::queue &queue, trsv_descr_t *p_trsv_descr);
ONEMKL_EXPORT sycl::event release_trsv_descr(sycl::queue &queue, trsv_descr_t trsv_descr,
                                             const std::vector<sycl::event> &dependencies = {});

// Temporary buffer size
ONEMKL_EXPORT sycl::event trsv_buffer_size(sycl::queue &queue, uplo uplo_val,
                                           transpose transpose_val, diag diag_val,
                                           matrix_handle_t A_handle, dense_vector_handle_t x,
                                           dense_vector_handle_t y, trsv_alg alg,
                                           trsv_descr_t trsv_descr, std::int64_t &temp_buffer_size,
                                           const std::vector<sycl::event> &dependencies = {});

ONEMKL_EXPORT sycl::event optimize_gemm(sycl::queue &queue, transpose transpose_A,
                                        matrix_handle_t handle,
                                        const std::vector<sycl::event> &dependencies = {});

ONEMKL_EXPORT sycl::event optimize_gemm(sycl::queue &queue, transpose transpose_A,
                                        transpose transpose_B, layout dense_matrix_layout,
                                        const std::int64_t columns, matrix_handle_t handle,
                                        const std::vector<sycl::event> &dependencies = {});

ONEMKL_EXPORT sycl::event optimize_gemv(sycl::queue &queue, transpose transpose_val,
                                        matrix_handle_t handle,
                                        const std::vector<sycl::event> &dependencies = {});

ONEMKL_EXPORT void optimize_trsv(sycl::queue &queue, uplo uplo_val, transpose transpose_val,
                                 diag diag_val, matrix_handle_t A_handle, trsv_alg alg,
                                 trsv_descr_t trsv_descr, std::int64_t temp_buffer_size,
                                 sycl::buffer<std::uint8_t, 1> temp_buffer);
ONEMKL_EXPORT sycl::event optimize_trsv(sycl::queue &queue, uplo uplo_val, transpose transpose_val,
                                        diag diag_val, matrix_handle_t A_handle, trsv_alg alg,
                                        trsv_descr_t trsv_descr, std::int64_t temp_buffer_size,
                                        void *temp_buffer,
                                        const std::vector<sycl::event> &dependencies = {});

template <typename fpType>
ONEMKL_EXPORT void gemv(sycl::queue &queue, transpose transpose_val, const fpType alpha,
                        matrix_handle_t A_handle, sycl::buffer<fpType, 1> &x, const fpType beta,
                        sycl::buffer<fpType, 1> &y);

template <typename fpType>
ONEMKL_EXPORT sycl::event gemv(sycl::queue &queue, transpose transpose_val, const fpType alpha,
                               matrix_handle_t A_handle, const fpType *x, const fpType beta,
                               fpType *y, const std::vector<sycl::event> &dependencies = {});

ONEMKL_EXPORT sycl::event trsv(sycl::queue &queue, uplo uplo_val, transpose transpose_val,
                               diag diag_val, matrix_handle_t A_handle, dense_vector_handle_t x,
                               dense_vector_handle_t y, trsv_alg alg, trsv_descr_t trsv_descr,
                               const std::vector<sycl::event> &dependencies = {});

template <typename fpType>
ONEMKL_EXPORT void gemm(sycl::queue &queue, layout dense_matrix_layout, transpose transpose_A,
                        transpose transpose_B, const fpType alpha, matrix_handle_t A_handle,
                        sycl::buffer<fpType, 1> &B, const std::int64_t columns,
                        const std::int64_t ldb, const fpType beta, sycl::buffer<fpType, 1> &C,
                        const std::int64_t ldc);

template <typename fpType>
ONEMKL_EXPORT sycl::event gemm(sycl::queue &queue, layout dense_matrix_layout,
                               transpose transpose_A, transpose transpose_B, const fpType alpha,
                               matrix_handle_t A_handle, const fpType *B,
                               const std::int64_t columns, const std::int64_t ldb,
                               const fpType beta, fpType *C, const std::int64_t ldc,
                               const std::vector<sycl::event> &dependencies = {});
