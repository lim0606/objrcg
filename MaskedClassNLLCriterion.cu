#include <THC/THC.h>
#include <THC/THCApply.cuh>
#include "common.h"

#include <stdio.h>
#include <assert.h>

static const int NTHREADS = 32;

__global__ void cunn_MaskedClassNLLCriterion_updateOutput_kernel1(float *output,
                                                           float *total_weight,
                                                           float *input,
                                                           float *target,
                                                           float *mask,
                                                           float *weights,
                                                           int size_average,
                                                           int n_classes) {
  assert(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0);

  // TODO: T4951791 Reuse code between updateOutput_kernel1 and
  // updateOutput_kernel.

  int t = (int)*target - 1;
  float m = mask ? (float)*mask : 1.0f;
  assert(t >= 0 && t < n_classes);
  float cur_weight = weights ? weights[t] : 1.0f; 
  cur_weight = cur_weight * m;
  *output = -cur_weight * input[t];
  *total_weight = cur_weight;
  if (size_average && *total_weight > 0) {
    *output /= *total_weight;
  }
}

__global__ void cunn_MaskedClassNLLCriterion_updateOutput_kernel(float *output,
                                                           float *total_weight,
                                                           float *input,
                                                           float *target,
                                                           float *mask,
                                                           float *weights,
                                                           int size_average,
                                                           int nframe,
                                                           int ndim,
                                                           int n_classes) {
  __shared__ float shInputs[NTHREADS], acc_weight[NTHREADS];
  int i, t;
  float m;
  float cur_weight;

  shInputs[threadIdx.x] = 0.0f;
  acc_weight[threadIdx.x] = 0.0f;
  for (i = threadIdx.x; i < nframe; i += NTHREADS) {
      t = target[i] - 1;
      m = mask ? mask[i] : 1.0f;
      assert(t >= 0 && t < n_classes);
      cur_weight = weights ? weights[t] : 1.0f;
      cur_weight = cur_weight * m;
      shInputs[threadIdx.x] -= input[i * ndim + t] * cur_weight;
      acc_weight[threadIdx.x] += cur_weight;
  }
  __syncthreads();

  // TODO: T4951791 Reuse code between updateOutput_kernel1 and
  // updateOutput_kernel

  if (threadIdx.x == 0) {
    *output = *total_weight = 0;
    for (i = 0; i < NTHREADS; ++i){
      *output += shInputs[i];
      *total_weight += acc_weight[i];
    }
    if (size_average && *total_weight > 0) {
      *output /= *total_weight;
    }
  }
}

__global__ void cunn_MaskedClassNLLCriterion_updateGradInput_kernel1(
  float* gradInput,
  float* weights,
  float* target,
  float* mask,
  float* total_weight,
  int size_average,
  int n_classes)
{
  if (*total_weight <= 0) {
    return;
  }
  float norm = size_average ? (1.0f / *total_weight) : 1.0f;
  int t = (int)*target - 1;
  float m = mask ? (float)*mask : 1.0f;
  assert(t >= 0 && t < n_classes);
  gradInput[t] = -(weights ? weights[t] : 1.0f) * norm * m;
}

__global__ void cunn_MaskedClassNLLCriterion_updateGradInput_kernel(
  float *gradInput,
  float *target,
  float *mask,
  float *weights,
  float *total_weight,
  int size_average,
  int nframe,
  int ndim,
  int n_classes)
{
  if (*total_weight <= 0) {
    return;
  }
  int i, t;
  float m;
  float norm = size_average ? (1.0f / *total_weight) : 1.0f;

  for (i = threadIdx.x; i < nframe; i += NTHREADS) {
    t = (int)target[i] - 1;
    m = mask ? (float)mask[i] : 1.0f;
    assert(t >= 0 && t < n_classes);
    gradInput[i * ndim + t] = -(weights ? weights[t] : 1.0f) * norm * m;
  }
}

extern "C"
void THNN_CudaMaskedClassNLLCriterion_updateOutput(
          THCState *state, 
          THCudaTensor *input, 
          THCudaTensor *target,
          THCudaTensor *mask, 
          THCudaTensor *output, 
          bool sizeAverage, 
          THCudaTensor *weights, 
          THCudaTensor *total_weight)
{
  if (THCudaTensor_nDimension(state, target) > 1) {
    THError("multi-target not supported");
  }

  int n_dims = THCudaTensor_nDimension(state, input);
  int n_classes = THCudaTensor_size(state, input, n_dims - 1);

  if (weights && mask)
    THCUNN_assertSameGPU(state, 6, input, target, mask, weights, output, total_weight);
  else if (weights)
    THCUNN_assertSameGPU(state, 5, input, target, weights, output, total_weight); 
  else if (mask) 
    THCUNN_assertSameGPU(state, 5, input, target, mask, output, total_weight);
  else 
    THCUNN_assertSameGPU(state, 4, input, target, output, total_weight);

  if (THCudaTensor_nDimension(state, input) > 2) {
    THArgCheck(0, 2, "vector or matrix expected");
  }

  input = THCudaTensor_newContiguous(state, input);
  weights = weights ? THCudaTensor_newContiguous(state, weights) : NULL;
  mask = mask ? THCudaTensor_newContiguous(state, mask) : NULL;
  target = THCudaTensor_newContiguous(state, target);

  float *input_data = THCudaTensor_data(state, input);
  float *weights_data = weights ? THCudaTensor_data(state, weights) : NULL;
  float *mask_data = mask ? THCudaTensor_data(state, mask) : NULL;
  float *target_data = THCudaTensor_data(state, target);
  float *output_data = THCudaTensor_data(state, output);
  float *total_weight_data = THCudaTensor_data(state, total_weight);

  if (THCudaTensor_nDimension(state, input) == 1) {
    cunn_MaskedClassNLLCriterion_updateOutput_kernel1
      <<<1, 1, 0, THCState_getCurrentStream(state)>>>(
        output_data,
        total_weight_data,
        input_data,
        target_data,
        mask_data,
        weights_data,
        sizeAverage,
        n_classes
    );

  } else if (THCudaTensor_nDimension(state, input) == 2) {
    cunn_MaskedClassNLLCriterion_updateOutput_kernel
      <<<1, NTHREADS, 0, THCState_getCurrentStream(state)>>>(
        output_data,
        total_weight_data,
        input_data,
        target_data,
        mask_data,
        weights_data,
        sizeAverage,
        THCudaTensor_size(state, input, 0),
        THCudaTensor_size(state, input, 1),
        n_classes
    );
  }
  THCudaCheck(cudaGetLastError());

  if (weights)
    THCudaTensor_free(state, weights);
  if (mask)
    THCudaTensor_free(state, mask);
  THCudaTensor_free(state, target);
  THCudaTensor_free(state, input);
}

extern "C"
void THNN_CudaMaskedClassNLLCriterion_updateGradInput(
          THCState *state, 
          THCudaTensor *input, 
          THCudaTensor *target,
          THCudaTensor *mask,
          THCudaTensor *gradInput,
          bool sizeAverage,
          THCudaTensor *weights,
          THCudaTensor *total_weight) {
  if (THCudaTensor_nDimension(state, target) > 1) {
    THError("multi-target not supported");
  }

  int n_dims = THCudaTensor_nDimension(state, input);
  int n_classes = THCudaTensor_size(state, input, n_dims - 1);

  THArgCheck(THCudaTensor_isContiguous(state, gradInput), 4, "gradInput must be contiguous");

  if (weights && mask)
    THCUNN_assertSameGPU(state, 6, weights, mask, input, target, gradInput, total_weight);
  else if (weights)
    THCUNN_assertSameGPU(state, 5, weights, input, target, gradInput, total_weight);
  else if (mask)
    THCUNN_assertSameGPU(state, 5, mask, input, target, gradInput, total_weight);
  else
    THCUNN_assertSameGPU(state, 4, input, target, gradInput, total_weight);

  if (THCudaTensor_nDimension(state, input) > 2) {
    THArgCheck(0, 2, "vector or matrix expected");
  }

  weights = weights ? THCudaTensor_newContiguous(state, weights) : NULL;
  mask = mask ? THCudaTensor_newContiguous(state, mask) : NULL;
  target = THCudaTensor_newContiguous(state, target);

  float *weights_data = weights ? THCudaTensor_data(state, weights) : NULL;
  float *mask_data = mask ? THCudaTensor_data(state, mask) : NULL;
  float *gradInput_data = THCudaTensor_data(state, gradInput);
  float *target_data = THCudaTensor_data(state, target);
  float *total_weight_data = THCudaTensor_data(state, total_weight);

  if (THCudaTensor_nDimension(state, input) == 1) {
    cunn_MaskedClassNLLCriterion_updateGradInput_kernel1
      <<<1, 1, 0, THCState_getCurrentStream(state)>>>(
        gradInput_data,
        weights_data,
        target_data,
        mask_data,
        total_weight_data,
        sizeAverage,
        n_classes
    );
  } else {
    cunn_MaskedClassNLLCriterion_updateGradInput_kernel
      <<<1, NTHREADS, 0, THCState_getCurrentStream(state)>>>(
        gradInput_data,
        target_data,
        mask_data,
        weights_data,
        total_weight_data,
        sizeAverage,
        THCudaTensor_size(state, input, 0),
        THCudaTensor_size(state, input, 1),
        n_classes
    );
  }
  THCudaCheck(cudaGetLastError());

  if (weights)
    THCudaTensor_free(state, weights);
  if (mask)
    THCudaTensor_free(state, mask);
  THCudaTensor_free(state, target);
}
