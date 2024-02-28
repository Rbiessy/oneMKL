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

#include "cusparse_error.hpp"
#include "cusparse_helper.hpp"
#include "cusparse_internal_containers.hpp"
#include "cusparse_task.hpp"
#include "sparse_blas/macros.hpp"

namespace oneapi::mkl::sparse::cusparse {

using namespace oneapi::mkl::sparse::detail;

/**
 * In this file CusparseScopedContextHandler are used to ensure that a cusparseHandle_t is created before any other cuSPARSE call, as required by the specification.
*/

// Dense vector
template <typename fpType>
void create_dense_vector(sycl::queue &queue, dense_vector_handle_t *p_dvhandle, std::int64_t size,
                         sycl::buffer<fpType, 1> &val) {
    auto event = queue.submit([&](sycl::handler &cgh) {
        auto acc = val.template get_access<sycl::access::mode::read_write>(cgh);
        detail::submit_host_task(cgh, queue, [=](detail::CusparseScopedContextHandler &sc) {
            // Ensure that a cusparse handle is created before any other cuSPARSE function is called.
            sc.get_handle(queue);
            auto cuda_type = detail::CudaEnumType<fpType>::value;
            cusparseDnVecDescr_t cu_dvhandle;
            auto status = cusparseCreateDnVec(&cu_dvhandle, size, sc.get_mem(acc), cuda_type);
            detail::check_status(status, "create_dense_vector");
            auto internal_dvhandle = new detail::dense_vector_handle(cu_dvhandle, val);
            *p_dvhandle = reinterpret_cast<dense_vector_handle_t>(internal_dvhandle);
        });
    });
    event.wait_and_throw();
}

template <typename fpType>
sycl::event create_dense_vector(sycl::queue &queue, dense_vector_handle_t *p_dvhandle,
                                std::int64_t size, fpType *val,
                                const std::vector<sycl::event> &dependencies) {
    return queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        detail::submit_host_task(cgh, queue, [=](detail::CusparseScopedContextHandler &sc) {
            // Ensure that a cusparse handle is created before any other cuSPARSE function is called.
            sc.get_handle(queue);
            auto cuda_type = detail::CudaEnumType<fpType>::value;
            cusparseDnVecDescr_t cu_dvhandle;
            auto status = cusparseCreateDnVec(&cu_dvhandle, size, sc.get_mem(val), cuda_type);
            detail::check_status(status, "create_dense_vector");
            auto internal_dvhandle = new detail::dense_vector_handle(cu_dvhandle, val);
            *p_dvhandle = reinterpret_cast<dense_vector_handle_t>(internal_dvhandle);
        });
    });
}

#define INSTANTIATE_CREATE_DENSE_VECTOR(FP_TYPE, FP_SUFFIX)                               \
    template std::enable_if_t<is_fp_supported_v<FP_TYPE>> create_dense_vector<FP_TYPE>(   \
        sycl::queue & queue, dense_vector_handle_t * p_dvhandle, std::int64_t size,       \
        sycl::buffer<FP_TYPE, 1> & val);                                                  \
    template std::enable_if_t<is_fp_supported_v<FP_TYPE>, sycl::event>                    \
    create_dense_vector<FP_TYPE>(sycl::queue & queue, dense_vector_handle_t * p_dvhandle, \
                                 std::int64_t size, FP_TYPE * val,                        \
                                 const std::vector<sycl::event> &dependencies)
FOR_EACH_FP_TYPE(INSTANTIATE_CREATE_DENSE_VECTOR);
#undef INSTANTIATE_CREATE_DENSE_VECTOR

// Dense matrix
template <typename fpType>
void create_dense_matrix(sycl::queue &queue, dense_matrix_handle_t *p_dmhandle,
                         std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,
                         layout dense_layout, sycl::buffer<fpType, 1> &val) {
    auto event = queue.submit([&](sycl::handler &cgh) {
        auto acc = val.template get_access<sycl::access::mode::read_write>(cgh);
        detail::submit_host_task(cgh, queue, [=](detail::CusparseScopedContextHandler &sc) {
            // Ensure that a cusparse handle is created before any other cuSPARSE function is called.
            sc.get_handle(queue);
            auto cuda_type = detail::CudaEnumType<fpType>::value;
            auto cuda_order = detail::get_cuda_order(dense_layout);
            cusparseDnMatDescr_t cu_dmhandle;
            auto status = cusparseCreateDnMat(&cu_dmhandle, num_rows, num_cols, ld, sc.get_mem(acc),
                                              cuda_type, cuda_order);
            detail::check_status(status, "create_dense_matrix");
            auto internal_dmhandle = new detail::dense_matrix_handle(cu_dmhandle, val);
            *p_dmhandle = reinterpret_cast<dense_matrix_handle_t>(internal_dmhandle);
        });
    });
    event.wait_and_throw();
}
template <typename fpType>
sycl::event create_dense_matrix(sycl::queue &queue, dense_matrix_handle_t *p_dmhandle,
                                std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld,
                                layout dense_layout, fpType *val,
                                const std::vector<sycl::event> &dependencies) {
    return queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        detail::submit_host_task(cgh, queue, [=](detail::CusparseScopedContextHandler &sc) {
            // Ensure that a cusparse handle is created before any other cuSPARSE function is called.
            sc.get_handle(queue);
            auto cuda_type = detail::CudaEnumType<fpType>::value;
            auto cuda_order = detail::get_cuda_order(dense_layout);
            cusparseDnMatDescr_t cu_dmhandle;
            auto status = cusparseCreateDnMat(&cu_dmhandle, num_rows, num_cols, ld, sc.get_mem(val),
                                              cuda_type, cuda_order);
            detail::check_status(status, "create_dense_matrix");
            auto internal_dmhandle = new detail::dense_matrix_handle(cu_dmhandle, val);
            *p_dmhandle = reinterpret_cast<dense_matrix_handle_t>(internal_dmhandle);
        });
    });
}

#define INSTANTIATE_CREATE_DENSE_MATRIX(FP_TYPE, FP_SUFFIX)                                     \
    template std::enable_if_t<is_fp_supported_v<FP_TYPE>> create_dense_matrix<FP_TYPE>(         \
        sycl::queue & queue, dense_matrix_handle_t * p_dmhandle, std::int64_t num_rows,         \
        std::int64_t num_cols, std::int64_t ld, layout dense_layout,                            \
        sycl::buffer<FP_TYPE, 1> & val);                                                        \
    template std::enable_if_t<is_fp_supported_v<FP_TYPE>, sycl::event>                          \
    create_dense_matrix<FP_TYPE>(sycl::queue & queue, dense_matrix_handle_t * p_dmhandle,       \
                                 std::int64_t num_rows, std::int64_t num_cols, std::int64_t ld, \
                                 layout dense_layout, FP_TYPE * val,                            \
                                 const std::vector<sycl::event> &dependencies)
FOR_EACH_FP_TYPE(INSTANTIATE_CREATE_DENSE_MATRIX);
#undef INSTANTIATE_CREATE_DENSE_MATRIX

// CSR matrix
template <typename fpType, typename intType>
void create_csr_matrix(sycl::queue &queue, matrix_handle_t *p_smhandle, std::int64_t num_rows,
                       std::int64_t num_cols, std::int64_t nnz, index_base index,
                       sycl::buffer<intType, 1> &row_ptr, sycl::buffer<intType, 1> &col_ind,
                       sycl::buffer<fpType, 1> &val) {
    auto event = queue.submit([&](sycl::handler &cgh) {
        auto row_acc = row_ptr.template get_access<sycl::access::mode::read_write>(cgh);
        auto col_acc = col_ind.template get_access<sycl::access::mode::read_write>(cgh);
        auto val_acc = val.template get_access<sycl::access::mode::read_write>(cgh);
        detail::submit_host_task(cgh, queue, [=](detail::CusparseScopedContextHandler &sc) {
            // Ensure that a cusparse handle is created before any other cuSPARSE function is called.
            sc.get_handle(queue);
            auto cuda_type = detail::CudaEnumType<fpType>::value;
            auto cudaIndexType = detail::CudaIndexEnumType<intType>::value;
            auto cudaIndexBase = detail::get_cuda_index_base(index);
            cusparseSpMatDescr_t cu_smhandle;
            auto status = cusparseCreateCsr(
                &cu_smhandle, num_rows, num_cols, nnz, sc.get_mem(row_acc), sc.get_mem(col_acc),
                sc.get_mem(val_acc), cudaIndexType, cudaIndexType, cudaIndexBase, cuda_type);
            detail::check_status(status, "create_csr_matrix");
            auto internal_smhandle =
                new detail::sparse_matrix_handle(cu_smhandle, val, row_ptr, col_ind);
            *p_smhandle = reinterpret_cast<matrix_handle_t>(internal_smhandle);
        });
    });
    event.wait_and_throw();
}

template <typename fpType, typename intType>
sycl::event create_csr_matrix(sycl::queue &queue, matrix_handle_t *p_smhandle,
                              std::int64_t num_rows, std::int64_t num_cols, std::int64_t nnz,
                              index_base index, intType *row_ptr, intType *col_ind, fpType *val,
                              const std::vector<sycl::event> &dependencies) {
    return queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        detail::submit_host_task(cgh, queue, [=](detail::CusparseScopedContextHandler &sc) {
            // Ensure that a cusparse handle is created before any other cuSPARSE function is called.
            sc.get_handle(queue);
            auto cuda_type = detail::CudaEnumType<fpType>::value;
            auto cudaIndexType = detail::CudaIndexEnumType<intType>::value;
            auto cudaIndexBase = detail::get_cuda_index_base(index);
            cusparseSpMatDescr_t cu_smhandle;
            auto status = cusparseCreateCsr(
                &cu_smhandle, num_rows, num_cols, nnz, sc.get_mem(row_ptr), sc.get_mem(col_ind),
                sc.get_mem(val), cudaIndexType, cudaIndexType, cudaIndexBase, cuda_type);
            detail::check_status(status, "create_csr_matrix");
            auto internal_smhandle =
                new detail::sparse_matrix_handle(cu_smhandle, val, row_ptr, col_ind);
            *p_smhandle = reinterpret_cast<matrix_handle_t>(internal_smhandle);
        });
    });
}

#define INSTANTIATE_CREATE_CSR_MATRIX(FP_TYPE, FP_SUFFIX, INT_TYPE, INT_SUFFIX)        \
    template std::enable_if_t<are_fp_int_supported_v<FP_TYPE, INT_TYPE>>               \
    create_csr_matrix<FP_TYPE, INT_TYPE>(                                              \
        sycl::queue & queue, matrix_handle_t * p_smhandle, std::int64_t num_rows,      \
        std::int64_t num_cols, std::int64_t nnz, index_base index,                     \
        sycl::buffer<INT_TYPE, 1> & row_ptr, sycl::buffer<INT_TYPE, 1> & col_ind,      \
        sycl::buffer<FP_TYPE, 1> & val);                                               \
    template std::enable_if_t<are_fp_int_supported_v<FP_TYPE, INT_TYPE>, sycl::event>  \
    create_csr_matrix<FP_TYPE, INT_TYPE>(                                              \
        sycl::queue & queue, matrix_handle_t * p_smhandle, std::int64_t num_rows,      \
        std::int64_t num_cols, std::int64_t nnz, index_base index, INT_TYPE * row_ptr, \
        INT_TYPE * col_ind, FP_TYPE * val, const std::vector<sycl::event> &dependencies)
FOR_EACH_FP_AND_INT_TYPE(INSTANTIATE_CREATE_CSR_MATRIX);
#undef INSTANTIATE_CREATE_CSR_MATRIX

// Destroy data types
sycl::event destroy_dense_vector(sycl::queue &queue, dense_vector_handle_t dvhandle,
                                 const std::vector<sycl::event> &dependencies) {
    return queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        cgh.host_task([=]() {
            auto internal_dvhandle = reinterpret_cast<detail::dense_vector_handle *>(dvhandle);
            auto cu_dvhandle = internal_dvhandle->cu_handle;
            auto status = cusparseDestroyDnVec(cu_dvhandle);
            detail::check_status(status, "destroy_dense_vector");
            delete internal_dvhandle;
        });
    });
}
sycl::event destroy_dense_matrix(sycl::queue &queue, dense_matrix_handle_t dmhandle,
                                 const std::vector<sycl::event> &dependencies) {
    return queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        cgh.host_task([=]() {
            auto internal_dmhandle = reinterpret_cast<detail::dense_matrix_handle *>(dmhandle);
            auto cu_dmhandle = internal_dmhandle->cu_handle;
            auto status = cusparseDestroyDnMat(cu_dmhandle);
            detail::check_status(status, "destroy_dense_matrix");
            delete internal_dmhandle;
        });
    });
}
sycl::event destroy_csr_matrix(sycl::queue &queue, matrix_handle_t smhandle,
                               const std::vector<sycl::event> &dependencies) {
    return queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        cgh.host_task([=]() {
            auto internal_smhandle = reinterpret_cast<detail::sparse_matrix_handle *>(smhandle);
            auto cu_smhandle = internal_smhandle->cu_handle;
            auto status = cusparseDestroySpMat(cu_smhandle);
            detail::check_status(status, "destroy_csr_matrix");
            delete internal_smhandle;
        });
    });
}

// Matrix property
void set_matrix_property(sycl::queue &, matrix_handle_t, matrix_property) {
    // No equivalent in cuSPARSE
}

} // namespace oneapi::mkl::sparse::cusparse
