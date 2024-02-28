/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Copyright (C) 2022 Heidelberg University, Engineering Mathematics and Computing Lab (EMCL) and Computing Centre (URZ)
*
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

#ifndef _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_TASKS_HPP_
#define _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_TASKS_HPP_

#include "cusparse_internal_containers.hpp"
#include "cusparse_scope_handle.hpp"

namespace oneapi::mkl::sparse::cusparse::detail {

template <typename T, typename Container>
auto get_value_accessor(sycl::handler &cgh, Container container) {
    auto buffer_ptr =
        reinterpret_cast<sycl::buffer<T, 1> *>(container.value_container.buffer_ptr.get());
    return buffer_ptr->template get_access<sycl::access::mode::read_write>(cgh);
}

template <typename T, typename... Ts>
auto get_fp_accessors(sycl::handler &cgh, Ts... containers) {
    return std::array<sycl::accessor<T, 1>, sizeof...(containers)>{ get_value_accessor<T>(
        cgh, containers)... };
}

template <typename T>
auto get_row_accessor(sycl::handler &cgh, sparse_matrix_handle smhandle) {
    auto buffer_ptr =
        reinterpret_cast<sycl::buffer<T, 1> *>(smhandle.row_container.buffer_ptr.get());
    return buffer_ptr->template get_access<sycl::access::mode::read_write>(cgh);
}

template <typename T>
auto get_col_accessor(sycl::handler &cgh, sparse_matrix_handle smhandle) {
    auto buffer_ptr =
        reinterpret_cast<sycl::buffer<T, 1> *>(smhandle.col_container.buffer_ptr.get());
    return buffer_ptr->template get_access<sycl::access::mode::read_write>(cgh);
}

template <typename T>
auto get_int_accessors(sycl::handler &cgh, sparse_matrix_handle smhandle) {
    // TODO(Romain): Support possibly multiple sparse_matrix_handle
    return std::array<sycl::accessor<T, 1>, 2>{ get_row_accessor<T>(cgh, smhandle),
                                                get_col_accessor<T>(cgh, smhandle) };
}

template <typename Functor, typename... CaptureAcc>
void submit_host_task(sycl::handler &cgh, sycl::queue &queue, Functor functor,
                      CaptureAcc... accessors) {
    // Only capture the accessors to ensure the dependencies are properly handled
    // The accessors's pointer have already been set to the native container types in previous functions
    cgh.host_task([functor, queue, accessors...](sycl::interop_handle ih) {
        auto unused = std::make_tuple(accessors...);
        (void)unused;
        auto sc = CusparseScopedContextHandler(queue, &ih);
        functor(sc);
    });
}

template <typename Functor, typename... Ts>
sycl::event dispatch_submit(sycl::queue queue, const std::vector<sycl::event> &dependencies,
                            Functor functor, sparse_matrix_handle sm_handle,
                            Ts... other_containers) {
    if (sm_handle.use_buffer()) {
        data_type value_type = sm_handle.get_value_type();
        data_type int_type = sm_handle.get_int_type();

#define ONEMKL_CUSPARSE_SUBMIT(FP_TYPE, INT_TYPE)                                      \
    return queue.submit([&](sycl::handler &cgh) {                                      \
        cgh.depends_on(dependencies);                                                  \
        auto fp_accs = get_fp_accessors<FP_TYPE>(cgh, sm_handle, other_containers...); \
        auto int_accs = get_int_accessors<INT_TYPE>(cgh, sm_handle);                   \
        submit_host_task(cgh, queue, functor, fp_accs, int_accs);                      \
    })
#define ONEMKL_CUSPARSE_SUBMIT_INT(FP_TYPE)            \
    if (int_type == data_type::int32) {                \
        ONEMKL_CUSPARSE_SUBMIT(FP_TYPE, std::int32_t); \
    }                                                  \
    else if (int_type == data_type::int64) {           \
        ONEMKL_CUSPARSE_SUBMIT(FP_TYPE, std::int64_t); \
    }

        if (value_type == data_type::real_fp32) {
            ONEMKL_CUSPARSE_SUBMIT_INT(float);
        }
        else if (value_type == data_type::real_fp64) {
            ONEMKL_CUSPARSE_SUBMIT_INT(double);
        }
        else if (value_type == data_type::complex_fp32) {
            ONEMKL_CUSPARSE_SUBMIT_INT(std::complex<float>);
        }
        else if (value_type == data_type::complex_fp64) {
            ONEMKL_CUSPARSE_SUBMIT_INT(std::complex<double>);
        }

#undef ONEMKL_CUSPARSE_SUBMIT_INT
#undef ONEMKL_CUSPARSE_SUBMIT

        throw oneapi::mkl::exception("sparse_blas", "dispatch_submit",
                                     "Could not dispatch buffer kernel to a supported type");
    }
    else {
        // USM submit does not need to capture accessors
        return queue.submit([&](sycl::handler &cgh) {
            cgh.depends_on(dependencies);
            submit_host_task(cgh, queue, functor);
        });
    }
}

} // namespace oneapi::mkl::sparse::cusparse::detail

#endif // _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_TASKS_HPP_
