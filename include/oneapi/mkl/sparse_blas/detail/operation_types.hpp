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

#ifndef _ONEMKL_SPARSE_BLAS_DETAIL_OPERATION_TYPES_HPP_
#define _ONEMKL_SPARSE_BLAS_DETAIL_OPERATION_TYPES_HPP_

namespace oneapi::mkl::sparse {

namespace detail {

// Each backend can create its own descriptor type or re-use the native descriptor types that will be reinterpret_cast'ed to the types below
struct trsv_descr;

} // namespace detail

typedef struct detail::trsv_descr *trsv_descr_t;

} // namespace oneapi::mkl::sparse

#endif // _ONEMKL_SPARSE_BLAS_DETAIL_OPERATION_TYPES_HPP_
