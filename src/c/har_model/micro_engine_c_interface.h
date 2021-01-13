// Copyright 2020 The MACE Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This is a generated file. DO NOT EDIT!

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void *har_cnn_GetMaceMicroEngineHandle();

bool har_cnn_RegisterInputData(void *handle, uint32_t idx,
                               const void *   input_buffer,
                               const int32_t *input_dims);

bool har_cnn_Interpret(void *handle);

bool har_cnn_GetInterpretResult(void *handle, const uint32_t idx,
                                void **output_data, const int32_t **output_dims,
                                uint32_t *output_dim_size);

#ifdef __cplusplus
}
#endif
