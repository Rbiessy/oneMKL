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
#ifndef _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_HELPER_HPP_
#define _ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_HELPER_HPP_

#include <complex>
#include <cstdint>
#include <limits>
#include <string>

#include <cusparse.h>

#include "oneapi/mkl/sparse_blas/types.hpp"
#include "cusparse_error.hpp"

namespace oneapi::mkl::sparse::cusparse::detail {

template <typename T>
struct CudaEnumType;
template <>
struct CudaEnumType<float> {
    static constexpr cudaDataType_t value = CUDA_R_32F;
};
template <>
struct CudaEnumType<double> {
    static constexpr cudaDataType_t value = CUDA_R_64F;
};
template <>
struct CudaEnumType<std::complex<float>> {
    static constexpr cudaDataType_t value = CUDA_C_32F;
};
template <>
struct CudaEnumType<std::complex<double>> {
    static constexpr cudaDataType_t value = CUDA_C_64F;
};

template <typename T>
struct CudaIndexEnumType;
template <>
struct CudaIndexEnumType<std::int32_t> {
    static constexpr cusparseIndexType_t value = CUSPARSE_INDEX_32I;
};
template <>
struct CudaIndexEnumType<std::int64_t> {
    static constexpr cusparseIndexType_t value = CUSPARSE_INDEX_64I;
};

template <typename E>
inline std::string enum_to_str(E e) {
    return std::to_string(static_cast<char>(e));
}

inline std::int64_t safe_cast(std::size_t x, const std::string& func_name) {
    if (x >= std::numeric_limits<std::int64_t>::max()) {
        throw oneapi::mkl::exception(
            "sparse_blas", func_name,
            "Integer overflow: " + std::to_string(x) + " does not fit in std::int64_t");
    }
    return static_cast<std::int64_t>(x);
}

inline cusparseOrder_t get_cuda_order(layout l) {
    switch (l) {
        case layout::row_major: return CUSPARSE_ORDER_ROW;
        case layout::col_major: return CUSPARSE_ORDER_COL;
        default:
            throw oneapi::mkl::invalid_argument("sparse_blas", "get_cuda_order",
                                                "Unknown layout: " + enum_to_str(l));
    }
}

inline cusparseIndexBase_t get_cuda_index_base(index_base index) {
    switch (index) {
        case index_base::zero: return CUSPARSE_INDEX_BASE_ZERO;
        case index_base::one: return CUSPARSE_INDEX_BASE_ONE;
        default:
            throw oneapi::mkl::invalid_argument("sparse_blas", "get_cuda_index_base",
                                                "Unknown index_base: " + enum_to_str(index));
    }
}

inline cusparseOperation_t get_cuda_operation(transpose op) {
    switch (op) {
        case transpose::nontrans: return CUSPARSE_OPERATION_NON_TRANSPOSE;
        case transpose::trans: return CUSPARSE_OPERATION_TRANSPOSE;
        case transpose::conjtrans: return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
        default:
            throw oneapi::mkl::invalid_argument("sparse_blas", "get_cuda_operation",
                                                "Unknown transpose operation: " + enum_to_str(op));
    }
}

inline auto get_cuda_uplo(uplo uplo_val) {
    switch (uplo_val) {
        case uplo::upper: return CUSPARSE_FILL_MODE_UPPER;
        case uplo::lower: return CUSPARSE_FILL_MODE_LOWER;
        default:
            throw oneapi::mkl::invalid_argument("sparse_blas", "get_cuda_uplo",
                                                "Unknown uplo: " + enum_to_str(uplo_val));
    }
}

inline auto get_cuda_diag(diag diag_val) {
    switch (diag_val) {
        case diag::nonunit: return CUSPARSE_DIAG_TYPE_NON_UNIT;
        case diag::unit: return CUSPARSE_DIAG_TYPE_UNIT;
        default:
            throw oneapi::mkl::invalid_argument("sparse_blas", "get_cuda_diag",
                                                "Unknown diag: " + enum_to_str(diag_val));
    }
}

inline void set_matrix_attributes(const std::string& func_name, cusparseSpMatDescr_t cu_a,
                                  uplo uplo_val, diag diag_val) {
    auto cu_fill_mode = get_cuda_uplo(uplo_val);
    auto status = cusparseSpMatSetAttribute(cu_a, CUSPARSE_SPMAT_FILL_MODE, &cu_fill_mode,
                                            sizeof(cu_fill_mode));
    check_status(status, func_name + "/set_uplo");

    auto cu_diag_type = get_cuda_diag(diag_val);
    status = cusparseSpMatSetAttribute(cu_a, CUSPARSE_SPMAT_DIAG_TYPE, &cu_diag_type,
                                       sizeof(cu_diag_type));
    check_status(status, func_name + "/set_diag");
}

} // namespace oneapi::mkl::sparse::cusparse::detail

#endif //_ONEMKL_SPARSE_BLAS_BACKENDS_CUSPARSE_HELPER_HPP_
