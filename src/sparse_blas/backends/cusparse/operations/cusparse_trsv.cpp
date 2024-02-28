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

namespace detail {

struct trsv_descr {
    cusparseSpSVDescr_t cu_descr;
    cudaDataType compute_type;

    // TODO(Romain): Make these static constexpr
    float alpha_one_fp32 = 1.f;
    double alpha_one_fp64 = 1.0;
    cuComplex alpha_one_cx32 = make_cuComplex(1.f, 0.f);
    cuDoubleComplex alpha_one_cx64 = make_cuDoubleComplex(1.0, 0.0);

    void *get_alpha() {
        switch (compute_type) {
            case CUDA_R_32F: return &alpha_one_fp32;
            case CUDA_R_64F: return &alpha_one_fp64;
            case CUDA_C_32F: return &alpha_one_cx32;
            case CUDA_C_64F: return &alpha_one_cx64;
            default: return nullptr;
        }
    }
};

inline auto get_cuda_trsv_alg(trsv_alg /*alg*/) {
    return CUSPARSE_SPSV_ALG_DEFAULT;
}

void optimize_trsv_impl(cusparseHandle_t cu_handle, uplo uplo_val, transpose transpose_val,
                        diag diag_val, matrix_handle_t A_handle, trsv_alg alg,
                        trsv_descr_t trsv_descr, void *temp_buffer_ptr) {
    auto cu_a = reinterpret_cast<detail::sparse_matrix_handle *>(A_handle)->cu_handle;
    detail::set_matrix_attributes("optimize_trsv", cu_a, uplo_val, diag_val);
    cusparseConstDnVecDescr_t cu_x = nullptr;
    cusparseDnVecDescr_t cu_y = nullptr;
    auto cu_op = detail::get_cuda_operation(transpose_val);
    auto internal_trsv_descr = reinterpret_cast<struct detail::trsv_descr *>(trsv_descr);
    auto cu_alpha = internal_trsv_descr->get_alpha();
    auto cu_type = internal_trsv_descr->compute_type;
    auto cu_alg = detail::get_cuda_trsv_alg(alg);
    auto cu_descr = internal_trsv_descr->cu_descr;
    auto status = cusparseSpSV_analysis(cu_handle, cu_op, cu_alpha, cu_a, cu_x, cu_y, cu_type,
                                        cu_alg, cu_descr, temp_buffer_ptr);
    detail::check_status(status, "optimize_trsv");
}

} // namespace detail

template <typename fpType>
void init_trsv_descr(sycl::queue &queue, trsv_descr_t *p_trsv_descr) {
    // Ensure that a cusparse handle is created before any other cuSPARSE function is called.
    detail::CusparseScopedContextHandler sc(queue);
    sc.get_handle(queue);

    auto internal_trsv_descr = new detail::trsv_descr();
    internal_trsv_descr->compute_type = detail::CudaEnumType<fpType>::value;
    auto status = cusparseSpSV_createDescr(&internal_trsv_descr->cu_descr);
    detail::check_status(status, "init_trsv_descr");
    *p_trsv_descr = reinterpret_cast<trsv_descr_t>(internal_trsv_descr);
}

#define INSTANTIATE_INIT_TRSV_DESCR(FP_TYPE, FP_SUFFIX) \
    template void init_trsv_descr<FP_TYPE>(sycl::queue & queue, trsv_descr_t * p_trsv_descr)
FOR_EACH_FP_TYPE(INSTANTIATE_INIT_TRSV_DESCR);
#undef INSTANTIATE_INIT_TRSV_DESCR

sycl::event release_trsv_descr(sycl::queue &queue, trsv_descr_t trsv_descr,
                               const std::vector<sycl::event> &dependencies) {
    return queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        cgh.host_task([=]() {
            auto internal_trsv_descr = reinterpret_cast<struct detail::trsv_descr *>(trsv_descr);
            auto status = cusparseSpSV_destroyDescr(internal_trsv_descr->cu_descr);
            detail::check_status(status, "release_trsv_descr");
            delete internal_trsv_descr;
        });
    });
}

sycl::event trsv_buffer_size(sycl::queue &queue, uplo uplo_val, transpose transpose_val,
                             diag diag_val, matrix_handle_t A_handle, dense_vector_handle_t x,
                             dense_vector_handle_t y, trsv_alg alg, trsv_descr_t trsv_descr,
                             std::int64_t &temp_buffer_size,
                             const std::vector<sycl::event> &dependencies) {
    auto internal_A = *reinterpret_cast<detail::sparse_matrix_handle *>(A_handle);
    auto internal_x = *reinterpret_cast<detail::dense_vector_handle *>(x);
    auto internal_y = *reinterpret_cast<detail::dense_vector_handle *>(y);
    detail::check_all_containers_compatible(__FUNCTION__, internal_A, internal_x, internal_y);
    auto functor = [=, &temp_buffer_size](detail::CusparseScopedContextHandler &sc) {
        auto cu_handle = sc.get_handle(queue);
        auto cu_a = internal_A.cu_handle;
        auto cu_x = internal_x.cu_handle;
        auto cu_y = internal_y.cu_handle;
        detail::set_matrix_attributes(__FUNCTION__, cu_a, uplo_val, diag_val);
        auto cu_op = detail::get_cuda_operation(transpose_val);
        auto internal_trsv_descr = reinterpret_cast<struct detail::trsv_descr *>(trsv_descr);
        auto cu_alpha = internal_trsv_descr->get_alpha();
        auto cu_type = internal_trsv_descr->compute_type;
        auto cu_alg = detail::get_cuda_trsv_alg(alg);
        auto cu_descr = internal_trsv_descr->cu_descr;
        std::size_t cu_buffer_size;
        auto status = cusparseSpSV_bufferSize(cu_handle, cu_op, cu_alpha, cu_a, cu_x, cu_y, cu_type,
                                              cu_alg, cu_descr, &cu_buffer_size);
        detail::check_status(status, __FUNCTION__);
        temp_buffer_size = detail::safe_cast(cu_buffer_size, __FUNCTION__);
    };
    return detail::dispatch_submit(queue, dependencies, functor, internal_A, internal_x,
                                   internal_y);
}

void optimize_trsv(sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,
                   matrix_handle_t A_handle, trsv_alg alg, trsv_descr_t trsv_descr,
                   std::int64_t /*temp_buffer_size*/, sycl::buffer<std::uint8_t, 1> temp_buffer) {
    auto event = queue.submit([&](sycl::handler &cgh) {
        // TODO(Romain): Get accessor from data handles
        auto temp_buffer_acc = temp_buffer.template get_access<sycl::access::mode::read_write>(cgh);
        detail::submit_host_task(cgh, queue, [=](detail::CusparseScopedContextHandler &sc) {
            auto cu_handle = sc.get_handle(queue);
            auto temp_buffer_ptr = sc.get_mem(temp_buffer_acc);
            detail::optimize_trsv_impl(cu_handle, uplo_val, transpose_val, diag_val, A_handle, alg,
                                       trsv_descr, temp_buffer_ptr);
        });
    });
    event.wait_and_throw();
}
sycl::event optimize_trsv(sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,
                          matrix_handle_t A_handle, trsv_alg alg, trsv_descr_t trsv_descr,
                          std::int64_t /*temp_buffer_size*/, void *temp_buffer,
                          const std::vector<sycl::event> &dependencies) {
    return queue.submit([&](sycl::handler &cgh) {
        cgh.depends_on(dependencies);
        detail::submit_host_task(cgh, queue, [=](detail::CusparseScopedContextHandler &sc) {
            auto cu_handle = sc.get_handle(queue);
            detail::optimize_trsv_impl(cu_handle, uplo_val, transpose_val, diag_val, A_handle, alg,
                                       trsv_descr, temp_buffer);
        });
    });
}

sycl::event trsv(sycl::queue &queue, uplo uplo_val, transpose transpose_val, diag diag_val,
                 matrix_handle_t A_handle, dense_vector_handle_t x, dense_vector_handle_t y,
                 trsv_alg alg, trsv_descr_t trsv_descr,
                 const std::vector<sycl::event> &dependencies) {
    auto internal_A = *reinterpret_cast<detail::sparse_matrix_handle *>(A_handle);
    auto internal_x = *reinterpret_cast<detail::dense_vector_handle *>(x);
    auto internal_y = *reinterpret_cast<detail::dense_vector_handle *>(y);
    detail::check_all_containers_compatible(__FUNCTION__, internal_A, internal_x, internal_y);
    auto functor = [=](detail::CusparseScopedContextHandler &sc) {
        auto cu_handle = sc.get_handle(queue);
        auto cu_a = internal_A.cu_handle;
        auto cu_x = internal_x.cu_handle;
        auto cu_y = internal_y.cu_handle;
        detail::set_matrix_attributes(__FUNCTION__, cu_a, uplo_val, diag_val);
        auto cu_op = detail::get_cuda_operation(transpose_val);
        auto internal_trsv_descr = reinterpret_cast<struct detail::trsv_descr *>(trsv_descr);
        auto cu_alpha = internal_trsv_descr->get_alpha();
        auto cu_type = internal_trsv_descr->compute_type;
        auto cu_alg = detail::get_cuda_trsv_alg(alg);
        auto cu_descr = internal_trsv_descr->cu_descr;
        auto status = cusparseSpSV_solve(cu_handle, cu_op, cu_alpha, cu_a, cu_x, cu_y, cu_type,
                                         cu_alg, cu_descr);
        detail::check_status(status, __FUNCTION__);
    };
    return detail::dispatch_submit(queue, dependencies, functor, internal_A, internal_x,
                                   internal_y);
}

} // namespace oneapi::mkl::sparse::cusparse
