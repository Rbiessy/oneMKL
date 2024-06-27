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
#include "sparse_blas/backends/cusparse/cusparse_handles.hpp"
#include "sparse_blas/common_op_verification.hpp"
#include "sparse_blas/macros.hpp"
#include "sparse_blas/sycl_helper.hpp"

namespace oneapi::mkl::sparse {

// Complete the definition of the incomplete type
struct spmm_descr {
    detail::generic_container workspace;
    std::size_t temp_buffer_size = 0;
};

} // namespace oneapi::mkl::sparse

namespace oneapi::mkl::sparse::cusparse {

void init_spmm_descr(sycl::queue& /*queue*/, spmm_descr_t* p_spmm_descr) {
    *p_spmm_descr = new spmm_descr();
}

sycl::event release_spmm_descr(sycl::queue& queue, spmm_descr_t spmm_descr,
                               const std::vector<sycl::event>& dependencies) {
    return detail::submit_release(queue, spmm_descr, dependencies);
}

inline auto get_cuda_spmm_alg(spmm_alg alg) {
    switch (alg) {
        case spmm_alg::coo_alg1: return CUSPARSE_SPMM_COO_ALG1;
        case spmm_alg::coo_alg2: return CUSPARSE_SPMM_COO_ALG2;
        case spmm_alg::coo_alg3: return CUSPARSE_SPMM_COO_ALG3;
        case spmm_alg::coo_alg4: return CUSPARSE_SPMM_COO_ALG4;
        case spmm_alg::csr_alg1: return CUSPARSE_SPMM_CSR_ALG1;
        case spmm_alg::csr_alg2: return CUSPARSE_SPMM_CSR_ALG2;
        case spmm_alg::csr_alg3: return CUSPARSE_SPMM_CSR_ALG3;
        default: return CUSPARSE_SPMM_ALG_DEFAULT;
    }
}

inline void fallback_alg_if_needed(oneapi::mkl::sparse::spmm_alg &alg, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB) {
    if (alg == oneapi::mkl::sparse::spmm_alg::csr_alg3 && (opA != oneapi::mkl::transpose::nontrans || opB == oneapi::mkl::transpose::conjtrans)) {
        // Avoid warnings printed on std::cerr
        alg = oneapi::mkl::sparse::spmm_alg::default_alg;
    }
}

void spmm_buffer_size(sycl::queue& queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                      const void* alpha, oneapi::mkl::sparse::matrix_view A_view,
                      oneapi::mkl::sparse::matrix_handle_t A_handle,
                      oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void* beta,
                      oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                      oneapi::mkl::sparse::spmm_alg alg,
                      oneapi::mkl::sparse::spmm_descr_t spmm_descr,
                      std::size_t& temp_buffer_size) {
    detail::check_valid_spmm_common(__FUNCTION__, queue, A_view, A_handle, B_handle, C_handle,
                                    alpha, beta);
    fallback_alg_if_needed(alg, opA, opB);
    auto functor = [=, &temp_buffer_size](CusparseScopedContextHandler& sc) {
        auto cu_handle = sc.get_handle(queue);
        auto cu_a = A_handle->backend_handle;
        auto cu_b = B_handle->backend_handle;
        auto cu_c = C_handle->backend_handle;
        auto cu_op_a = get_cuda_operation(opA);
        auto cu_op_b = get_cuda_operation(opB);
        auto cu_type = get_cuda_value_type(A_handle->value_container.data_type);
        auto cu_alg = get_cuda_spmm_alg(alg);
        auto status = cusparseSpMM_bufferSize(cu_handle, cu_op_a, cu_op_b, alpha, cu_a, cu_b, beta,
                                              cu_c, cu_type, cu_alg, &temp_buffer_size);
        check_status(status, __FUNCTION__);
    };
    auto event = dispatch_submit(__FUNCTION__, queue, functor, A_handle, B_handle, C_handle);
    event.wait_and_throw();
    spmm_descr->temp_buffer_size = temp_buffer_size;
}

void spmm_optimize_impl(cusparseHandle_t cu_handle, oneapi::mkl::transpose opA,
                        oneapi::mkl::transpose opB, const void* alpha,
                        oneapi::mkl::sparse::matrix_handle_t A_handle,
                        oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void* beta,
                        oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                        oneapi::mkl::sparse::spmm_alg alg, void* workspace_ptr) {
    auto cu_a = A_handle->backend_handle;
    auto cu_b = B_handle->backend_handle;
    auto cu_c = C_handle->backend_handle;
    auto cu_op_a = get_cuda_operation(opA);
    auto cu_op_b = get_cuda_operation(opB);
    auto cu_type = get_cuda_value_type(A_handle->value_container.data_type);
    auto cu_alg = get_cuda_spmm_alg(alg);
    auto status = cusparseSpMM_preprocess(cu_handle, cu_op_a, cu_op_b, alpha, cu_a, cu_b, beta,
                                          cu_c, cu_type, cu_alg, workspace_ptr);
    check_status(status, "optimize_spmm");
}

void spmm_optimize(sycl::queue& queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                   const void* alpha, oneapi::mkl::sparse::matrix_view A_view,
                   oneapi::mkl::sparse::matrix_handle_t A_handle,
                   oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void* beta,
                   oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                   oneapi::mkl::sparse::spmm_alg alg, oneapi::mkl::sparse::spmm_descr_t spmm_descr,
                   sycl::buffer<std::uint8_t, 1> workspace) {
    detail::check_valid_spmm_common(__FUNCTION__, queue, A_view, A_handle, B_handle, C_handle,
                                    alpha, beta);
    if (!A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__FUNCTION__);
    }
    // Copy the buffer to extend its lifetime until the descriptor is free'd.
    spmm_descr->workspace.set_buffer_untyped(workspace);
    if (alg == oneapi::mkl::sparse::spmm_alg::no_optimize_alg || workspace.size() == 0) {
        // cusparseSpMM_preprocess cannot be called if the workspace is empty
        return;
    }
    fallback_alg_if_needed(alg, opA, opB);
    auto functor = [=](CusparseScopedContextHandler& sc,
                       sycl::accessor<std::uint8_t> workspace_acc) {
        auto cu_handle = sc.get_handle(queue);
        auto workspace_ptr = sc.get_mem(workspace_acc);
        spmm_optimize_impl(cu_handle, opA, opB, alpha, A_handle, B_handle, beta, C_handle, alg,
                           workspace_ptr);
    };

    sycl::accessor<std::uint8_t, 1> workspace_placeholder_acc(workspace);
    auto event = dispatch_submit(__FUNCTION__, queue, functor, A_handle, workspace_placeholder_acc,
                                 B_handle, C_handle);
    event.wait_and_throw();
}

sycl::event spmm_optimize(sycl::queue& queue, oneapi::mkl::transpose opA,
                          oneapi::mkl::transpose opB, const void* alpha,
                          oneapi::mkl::sparse::matrix_view A_view,
                          oneapi::mkl::sparse::matrix_handle_t A_handle,
                          oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void* beta,
                          oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                          oneapi::mkl::sparse::spmm_alg alg,
                          oneapi::mkl::sparse::spmm_descr_t spmm_descr, void* workspace,
                          const std::vector<sycl::event>& dependencies) {
    detail::check_valid_spmm_common(__FUNCTION__, queue, A_view, A_handle, B_handle, C_handle,
                                    alpha, beta);
    if (A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__FUNCTION__);
    }
    spmm_descr->workspace.usm_ptr = workspace;
    if (alg == oneapi::mkl::sparse::spmm_alg::no_optimize_alg || workspace == nullptr) {
        // cusparseSpMM_preprocess cannot be called if the workspace is empty
        return detail::collapse_dependencies(queue, dependencies);
    }
    fallback_alg_if_needed(alg, opA, opB);
    auto functor = [=](CusparseScopedContextHandler& sc) {
        auto cu_handle = sc.get_handle(queue);
        spmm_optimize_impl(cu_handle, opA, opB, alpha, A_handle, B_handle, beta, C_handle, alg,
                           workspace);
    };

    return dispatch_submit(__FUNCTION__, queue, dependencies, functor, A_handle, B_handle,
                           C_handle);
}

sycl::event spmm(sycl::queue& queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                 const void* alpha, oneapi::mkl::sparse::matrix_view A_view,
                 oneapi::mkl::sparse::matrix_handle_t A_handle,
                 oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void* beta,
                 oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                 oneapi::mkl::sparse::spmm_alg alg, oneapi::mkl::sparse::spmm_descr_t spmm_descr,
                 const std::vector<sycl::event>& dependencies) {
    detail::check_valid_spmm_common(__FUNCTION__, queue, A_view, A_handle, B_handle, C_handle,
                                    alpha, beta);
    if (A_handle->all_use_buffer() != spmm_descr->workspace.use_buffer()) {
        detail::throw_incompatible_container(__FUNCTION__);
    }
    fallback_alg_if_needed(alg, opA, opB);
    if (A_handle->all_use_buffer() && spmm_descr->temp_buffer_size > 0) {
        // The accessor can only be bound to the cgh if the buffer size is
        // greater than 0
        auto functor = [=](CusparseScopedContextHandler& sc,
                           sycl::accessor<std::uint8_t> workspace_acc) {
            auto [cu_handle, cu_stream] = sc.get_handle_and_stream(queue);
            auto workspace_ptr = sc.get_mem(workspace_acc);
            auto cu_a = A_handle->backend_handle;
            auto cu_b = B_handle->backend_handle;
            auto cu_c = C_handle->backend_handle;
            auto cu_op_a = get_cuda_operation(opA);
            auto cu_op_b = get_cuda_operation(opB);
            auto cu_type = get_cuda_value_type(A_handle->value_container.data_type);
            auto cu_alg = get_cuda_spmm_alg(alg);
            auto status = cusparseSpMM(cu_handle, cu_op_a, cu_op_b, alpha, cu_a, cu_b, beta, cu_c,
                                       cu_type, cu_alg, workspace_ptr);
            check_status(status, __FUNCTION__);
            CUDA_ERROR_FUNC(cuStreamSynchronize, cu_stream);
        };
        sycl::accessor<std::uint8_t, 1> workspace_placeholder_acc(
            spmm_descr->workspace.get_buffer<std::uint8_t>());
        return dispatch_submit<true>(__FUNCTION__, queue, dependencies, functor, A_handle,
                                     workspace_placeholder_acc, B_handle, C_handle);
    }
    else {
        // The same dispatch_submit can be used for USM or buffers if no
        // workspace accessor is needed, workspace_ptr will be a nullptr in the
        // latter case.
        auto workspace_ptr = spmm_descr->workspace.usm_ptr;
        auto functor = [=](CusparseScopedContextHandler& sc) {
            auto [cu_handle, cu_stream] = sc.get_handle_and_stream(queue);
            auto cu_a = A_handle->backend_handle;
            auto cu_b = B_handle->backend_handle;
            auto cu_c = C_handle->backend_handle;
            auto cu_op_a = get_cuda_operation(opA);
            auto cu_op_b = get_cuda_operation(opB);
            auto cu_type = get_cuda_value_type(A_handle->value_container.data_type);
            auto cu_alg = get_cuda_spmm_alg(alg);
            auto status = cusparseSpMM(cu_handle, cu_op_a, cu_op_b, alpha, cu_a, cu_b, beta, cu_c,
                                       cu_type, cu_alg, workspace_ptr);
            check_status(status, __FUNCTION__);
            CUDA_ERROR_FUNC(cuStreamSynchronize, cu_stream);
        };
        return dispatch_submit(__FUNCTION__, queue, dependencies, functor, A_handle, B_handle,
                               C_handle);
    }
}

} // namespace oneapi::mkl::sparse::cusparse
