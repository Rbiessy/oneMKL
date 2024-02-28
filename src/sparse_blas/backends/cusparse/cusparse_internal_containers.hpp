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

#ifndef _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_INTERNAL_CONTAINERS_HPP_
#define _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_INTERNAL_CONTAINERS_HPP_

#include <memory>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <cusparse.h>

namespace oneapi::mkl::sparse::cusparse::detail {

enum data_type { int32, int64, real_fp32, real_fp64, complex_fp32, complex_fp64 };

inline std::string data_type_to_str(data_type data_type) {
    switch (data_type) {
        case int32: return "int32";
        case int64: return "int64";
        case real_fp32: return "real_fp32";
        case real_fp64: return "real_fp64";
        case complex_fp32: return "complex_fp32";
        case complex_fp64: return "complex_fp64";
        default: return "unknown";
    }
}

template <typename T>
data_type get_data_type() {
    if constexpr (std::is_same_v<T, std::int32_t>) {
        return data_type::int32;
    }
    else if constexpr (std::is_same_v<T, std::int64_t>) {
        return data_type::int64;
    }
    else if constexpr (std::is_same_v<T, float>) {
        return data_type::real_fp32;
    }
    else if constexpr (std::is_same_v<T, double>) {
        return data_type::real_fp64;
    }
    else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return data_type::complex_fp32;
    }
    else if constexpr (std::is_same_v<T, std::complex<double>>) {
        return data_type::complex_fp64;
    }
    else {
        static_assert(false, "Unsupported type");
    }
}

/**
 * Represent a non-templated container for USM or buffer.
*/
struct generic_container {
    // Store the buffer to properly handle the dependencies when the handle is used. This is not needed for USM pointers.
    // Use a void* type for the buffer to avoid using template arguments in every function using data handles.
    // Using reinterpret does not solve the issue as the returned buffer needs the type of the original buffer for the aligned_allocator.
    std::shared_ptr<void> buffer_ptr;

    // Underlying USM or buffer data type
    data_type data_type;

    template <typename T>
    generic_container(T* /*ptr*/) : buffer_ptr(),
                                    data_type(get_data_type<T>()) {}

    template <typename T>
    generic_container(const sycl::buffer<T, 1>& buffer)
            : buffer_ptr(std::make_shared<sycl::buffer<T, 1>>(buffer)),
              data_type(get_data_type<T>()) {}
};

template <typename CuHandle>
struct dense_handle {
    CuHandle cu_handle;

    generic_container value_container;

    template <typename T>
    dense_handle(CuHandle cu_handle, T* ptr)
            : cu_handle(cu_handle),
              value_container(generic_container(ptr)) {}

    template <typename T>
    dense_handle(CuHandle cu_handle, const sycl::buffer<T, 1>& value_buffer)
            : cu_handle(cu_handle),
              value_container(value_buffer) {}

    bool use_buffer() const {
        return static_cast<bool>(value_container.buffer_ptr);
    }

    data_type get_value_type() const {
        return value_container.data_type;
    }
};

template <typename CuHandle>
struct sparse_handle {
    CuHandle cu_handle;

    generic_container value_container;
    generic_container row_container;
    generic_container col_container;

    template <typename fpType, typename intType>
    sparse_handle(CuHandle cu_handle, fpType* value_ptr, intType* row_ptr, intType* col_ptr)
            : cu_handle(cu_handle),
              value_container(generic_container(value_ptr)),
              row_container(generic_container(row_ptr)),
              col_container(generic_container(col_ptr)) {}

    template <typename fpType, typename intType>
    sparse_handle(CuHandle cu_handle, const sycl::buffer<fpType, 1>& value_buffer,
                  const sycl::buffer<intType, 1>& row_buffer,
                  const sycl::buffer<intType, 1>& col_buffer)
            : cu_handle(cu_handle),
              value_container(value_buffer),
              row_container(row_buffer),
              col_container(col_buffer) {}

    bool use_buffer() const {
        return static_cast<bool>(value_container.buffer_ptr);
    }

    data_type get_value_type() const {
        return value_container.data_type;
    }

    data_type get_int_type() const {
        return row_container.data_type;
    }
};

using dense_vector_handle = dense_handle<cusparseDnVecDescr_t>;
using dense_matrix_handle = dense_handle<cusparseDnMatDescr_t>;
using sparse_matrix_handle = sparse_handle<cusparseSpMatDescr_t>;

/**
 * Check that all internal containers use the same container.
*/
template <typename... Ts>
void check_all_containers_use_buffers(const std::string& function,
                                      sparse_matrix_handle first_internal_container,
                                      Ts... internal_containers) {
    bool first_use_buffer = first_internal_container.use_buffer();
    for (const auto internal_container : { internal_containers... }) {
        if (internal_container.use_buffer() != first_use_buffer) {
            throw oneapi::mkl::invalid_argument(
                "sparse_blas", function,
                "Incompatible container types. All inputs and outputs must use the same container: buffer or USM");
        }
    }
}

/**
 * Check that all internal containers use the same container type, data type and integer type.
*/
template <typename... Ts>
void check_all_containers_compatible(const std::string& function,
                                     sparse_matrix_handle first_internal_container,
                                     Ts... internal_containers) {
    check_all_containers_use_buffers(function, first_internal_container, internal_containers...);
    data_type first_value_type = first_internal_container.get_value_type();
    data_type first_int_type = first_internal_container.get_int_type();
    for (const auto internal_container : { internal_containers... }) {
        const data_type other_value_type = internal_container.get_value_type();
        if (other_value_type != first_value_type) {
            throw oneapi::mkl::invalid_argument(
                "sparse_blas", function,
                "Incompatible data types expected " + data_type_to_str(first_value_type) +
                    " but got " + data_type_to_str(other_value_type));
        }
        if constexpr (std::is_same_v<decltype(internal_container), sparse_matrix_handle>) {
            const data_type other_int_type = internal_container.get_int_type();
            if (other_int_type != first_int_type) {
                throw oneapi::mkl::invalid_argument(
                    "sparse_blas", function,
                    "Incompatible integer types expected " + data_type_to_str(first_int_type) +
                        " but got " + data_type_to_str(other_int_type));
            }
        }
    }
}

} // namespace oneapi::mkl::sparse::cusparse::detail

#endif // _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_INTERNAL_CONTAINERS_HPP_
