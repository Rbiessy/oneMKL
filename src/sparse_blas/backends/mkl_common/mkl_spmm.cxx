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

// The operation descriptor is not needed as long as the backend does not have an equivalent type and does not support external workspace.
using spmm_descr = void *;

void init_spmm_descr(sycl::queue & /*queue*/, oneapi::mkl::sparse::spmm_descr_t *p_spmm_descr) {
    *p_spmm_descr = nullptr;
}

sycl::event release_spmm_descr(sycl::queue &queue, oneapi::mkl::sparse::spmm_descr_t /*spmm_descr*/,
                               const std::vector<sycl::event> &dependencies) {
    return detail::collapse_dependencies(queue, dependencies);
}

void check_valid_spmm(const std::string &function_name, oneapi::mkl::transpose opA,
                      oneapi::mkl::sparse::matrix_view A_view,
                      oneapi::mkl::sparse::matrix_handle_t A_handle,
                      oneapi::mkl::sparse::dense_matrix_handle_t B_handle,
                      oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                      bool is_alpha_host_accessible, bool is_beta_host_accessible) {
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    detail::check_valid_spmm_common(function_name, A_view, internal_A_handle, B_handle, C_handle,
                                    is_alpha_host_accessible, is_beta_host_accessible);

#if BACKEND == gpu
    detail::data_type data_type = internal_A_handle->get_value_type();
    if ((data_type == detail::data_type::complex_fp32 ||
         data_type == detail::data_type::complex_fp64) &&
        opA == oneapi::mkl::transpose::conjtrans &&
        internal_A_handle->has_matrix_property(oneapi::mkl::sparse::matrix_property::symmetric)) {
        throw mkl::unimplemented(
            "sparse_blas", function_name,
            "The backend does not support spmm using conjtrans and the symmetric property.");
    }
#else
    (void)opA;
#endif // BACKEND
}

void spmm_buffer_size(sycl::queue &queue, oneapi::mkl::transpose opA,
                      oneapi::mkl::transpose /*opB*/, const void *alpha,
                      oneapi::mkl::sparse::matrix_view A_view,
                      oneapi::mkl::sparse::matrix_handle_t A_handle,
                      oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void *beta,
                      oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                      oneapi::mkl::sparse::spmm_alg /*alg*/,
                      oneapi::mkl::sparse::spmm_descr_t /*spmm_descr*/,
                      std::size_t &temp_buffer_size) {
    // TODO: Add support for external workspace once the close-source oneMKL backend supports it.
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    check_valid_spmm(__func__, opA, A_view, A_handle, B_handle, C_handle, is_alpha_host_accessible,
                     is_beta_host_accessible);
    temp_buffer_size = 0;
}

void spmm_optimize(sycl::queue &queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose /*opB*/,
                   const void *alpha, oneapi::mkl::sparse::matrix_view A_view,
                   oneapi::mkl::sparse::matrix_handle_t A_handle,
                   oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void *beta,
                   oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                   oneapi::mkl::sparse::spmm_alg alg,
                   oneapi::mkl::sparse::spmm_descr_t /*spmm_descr*/,
                   sycl::buffer<std::uint8_t, 1> /*workspace*/) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    check_valid_spmm(__func__, opA, A_view, A_handle, B_handle, C_handle, is_alpha_host_accessible,
                     is_beta_host_accessible);
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    if (!internal_A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    if (alg == oneapi::mkl::sparse::spmm_alg::no_optimize_alg) {
        return;
    }
    internal_A_handle->can_be_reset = false;
    // TODO: Add support for spmm_optimize once the close-source oneMKL backend supports it.
}

sycl::event spmm_optimize(sycl::queue &queue, oneapi::mkl::transpose opA,
                          oneapi::mkl::transpose /*opB*/, const void *alpha,
                          oneapi::mkl::sparse::matrix_view A_view,
                          oneapi::mkl::sparse::matrix_handle_t A_handle,
                          oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void *beta,
                          oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                          oneapi::mkl::sparse::spmm_alg alg,
                          oneapi::mkl::sparse::spmm_descr_t /*spmm_descr*/, void * /*workspace*/,
                          const std::vector<sycl::event> &dependencies) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    check_valid_spmm(__func__, opA, A_view, A_handle, B_handle, C_handle, is_alpha_host_accessible,
                     is_beta_host_accessible);
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    if (internal_A_handle->all_use_buffer()) {
        detail::throw_incompatible_container(__func__);
    }
    if (alg == oneapi::mkl::sparse::spmm_alg::no_optimize_alg) {
        return detail::collapse_dependencies(queue, dependencies);
    }
    internal_A_handle->can_be_reset = false;
    // TODO: Add support for spmm_optimize once the close-source oneMKL backend supports it.
    return detail::collapse_dependencies(queue, dependencies);
}

template <typename T>
sycl::event internal_spmm(
    sycl::queue &queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB, const void *alpha,
    oneapi::mkl::sparse::matrix_view /*A_view*/, oneapi::mkl::sparse::matrix_handle_t A_handle,
    oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void *beta,
    oneapi::mkl::sparse::dense_matrix_handle_t C_handle, oneapi::mkl::sparse::spmm_alg /*alg*/,
    oneapi::mkl::sparse::spmm_descr_t /*spmm_descr*/, const std::vector<sycl::event> &dependencies,
    bool is_alpha_host_accessible, bool is_beta_host_accessible) {
    T host_alpha =
        detail::get_scalar_on_host(queue, static_cast<const T *>(alpha), is_alpha_host_accessible);
    T host_beta =
        detail::get_scalar_on_host(queue, static_cast<const T *>(beta), is_beta_host_accessible);
    auto internal_A_handle = detail::get_internal_handle(A_handle);
    internal_A_handle->can_be_reset = false;
    auto layout = B_handle->dense_layout;
    auto columns = C_handle->num_cols;
    auto ldb = B_handle->ld;
    auto ldc = C_handle->ld;
    if (internal_A_handle->all_use_buffer()) {
        oneapi::mkl::sparse::gemm(queue, layout, opA, opB, host_alpha,
                                  internal_A_handle->backend_handle, B_handle->get_buffer<T>(),
                                  columns, ldb, host_beta, C_handle->get_buffer<T>(), ldc);
        // Dependencies are not used for buffers
        return {};
    }
    else {
        return oneapi::mkl::sparse::gemm(queue, layout, opA, opB, host_alpha,
                                         internal_A_handle->backend_handle,
                                         B_handle->get_usm_ptr<T>(), columns, ldb, host_beta,
                                         C_handle->get_usm_ptr<T>(), ldc, dependencies);
    }
}

sycl::event spmm(sycl::queue &queue, oneapi::mkl::transpose opA, oneapi::mkl::transpose opB,
                 const void *alpha, oneapi::mkl::sparse::matrix_view A_view,
                 oneapi::mkl::sparse::matrix_handle_t A_handle,
                 oneapi::mkl::sparse::dense_matrix_handle_t B_handle, const void *beta,
                 oneapi::mkl::sparse::dense_matrix_handle_t C_handle,
                 oneapi::mkl::sparse::spmm_alg alg, oneapi::mkl::sparse::spmm_descr_t spmm_descr,
                 const std::vector<sycl::event> &dependencies) {
    bool is_alpha_host_accessible = detail::is_ptr_accessible_on_host(queue, alpha);
    bool is_beta_host_accessible = detail::is_ptr_accessible_on_host(queue, beta);
    check_valid_spmm(__func__, opA, A_view, A_handle, B_handle, C_handle, is_alpha_host_accessible,
                     is_beta_host_accessible);
    auto value_type = detail::get_internal_handle(A_handle)->get_value_type();
    DISPATCH_MKL_OPERATION("spmm", value_type, internal_spmm, queue, opA, opB, alpha, A_view,
                           A_handle, B_handle, beta, C_handle, alg, spmm_descr, dependencies,
                           is_alpha_host_accessible, is_beta_host_accessible);
}