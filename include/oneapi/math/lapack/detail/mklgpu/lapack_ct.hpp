/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#pragma once

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <complex>
#include <cstdint>

#include "oneapi/math/types.hpp"
#include "oneapi/math/lapack/types.hpp"
#include "oneapi/math/detail/backend_selector.hpp"
#include "oneapi/math/lapack/detail/mklgpu/onemath_lapack_mklgpu.hpp"

namespace oneapi {
namespace mkl {
namespace lapack {

#define LAPACK_BACKEND mklgpu
#include "oneapi/math/lapack/detail/mkl_common/lapack_ct.hxx"
#undef LAPACK_BACKEND

} //namespace lapack
} //namespace mkl
} //namespace oneapi