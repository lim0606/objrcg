#include <THC/THC.h>
#include <THC/THCApply.cuh>
#include "common.h"

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#if CUDA_VERSION >= 7000
#include <thrust/system/cuda/execution_policy.h>
#endif

#define EPS 1e-12

struct smoothl1_functor
{
  smoothl1_functor() {}

  __host__ __device__ float operator()(const float &x, const float &y) const
  {
    float z = fabsf(x-y);
    return z < 1.f ? 0.5f*z*z : z - 0.5f;
  }
};

struct smoothl1_w_mask_functor
{
  smoothl1_w_mask_functor() {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t)
  {
    float m = thrust::get<2>(t);
    float z = fabsf(thrust::get<0>(t) - thrust::get<1>(t));
    thrust::get<3>(t) =  m * (z < 1.f ? 0.5f*z*z : z - 0.5f);
  }
};

extern "C"
void THNN_CudaMaskedSmoothL1Criterion_updateOutput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *mask, THCudaTensor *output, bool sizeAverage)
{
  if (mask)
    THCUNN_assertSameGPU(state, 3, input, target, mask);
  else
    THCUNN_assertSameGPU(state, 2, input, target);

  THArgCheck(
    THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2,
    "input and target need to have the same number of elements"
  );
  if (mask)
    THArgCheck(
      THCudaTensor_nElement(state, mask) == THCudaTensor_nElement(state, target), 2,
      "mask and target need to have the same number of elements"
    );

  long size = THCudaTensor_nElement(state, input);

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);
  mask = mask ? THCudaTensor_newContiguous(state, mask) : NULL;
  THCudaTensor *buffer = mask ? THCudaTensor_newClone(state, mask) : NULL;
  if (buffer)
    THCudaTensor_fill(state, buffer, 0);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  thrust::device_ptr<float> mask_data(mask ? THCudaTensor_data(state, mask) : 0);
  thrust::device_ptr<float> buffer_data(buffer ? THCudaTensor_data(state, buffer) : 0); 

  float sum = 0;
  if (mask) {
    thrust::for_each(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      thrust::make_zip_iterator(thrust::make_tuple(input_data,      target_data,      mask_data,      buffer_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size, mask_data+size, buffer_data+size)),
      smoothl1_w_mask_functor()
    );
    sum = thrust::reduce(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      buffer_data, buffer_data+size, (float) 0,
      thrust::plus<float>()
    );
  }
  else { 
    sum = thrust::inner_product(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      input_data, input_data+size, target_data, (float) 0,
      thrust::plus<float>(), smoothl1_functor()
    );
  }

  if (sizeAverage && mask)
    sum /= (THCudaTensor_sumall(state, mask)+EPS);
  else if (sizeAverage)
    sum /= size;

  if (mask)
    THCudaTensor_free(state, mask);
  if (buffer)
    THCudaTensor_free(state, buffer);
  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);

  THCudaTensor_set1d(state, output, 0, sum);
}

struct smoothl1_updateGradInput_functor
{
  const float norm;

  smoothl1_updateGradInput_functor(float norm_)
    : norm(norm_)
  {}

  __host__ __device__ float operator()(const float &x, const float &y) const
  {
    float z = x - y;
    if (z < -1.f)
      return -norm;
    else if (z > 1.f)
      return norm;
    else
      return norm * z;
  }
};

struct smoothl1_w_mask_updateGradInput_functor
{
  const float norm;
  smoothl1_w_mask_updateGradInput_functor(float norm_)
    : norm(norm_)
  {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t)
  {
    float m = thrust::get<2>(t); // mask
    float z = thrust::get<0>(t) - thrust::get<1>(t);        // gradInput

    if (z < -1.f)
      thrust::get<3>(t) = -norm * m;
    else if (z > 1.f)
      thrust::get<3>(t) = norm * m;
    else
      thrust::get<3>(t) = norm * z * m;
  }
};

extern "C"
void THNN_CudaMaskedSmoothL1Criterion_updateGradInput(THCState *state, THCudaTensor *input, THCudaTensor *target, THCudaTensor *mask, THCudaTensor *gradInput, bool sizeAverage)
{
  if (mask)
    THCUNN_assertSameGPU(state, 4, input, target, mask, gradInput);
  else
    THCUNN_assertSameGPU(state, 3, input, target, gradInput);

  THArgCheck(
    THCudaTensor_nElement(state, input) == THCudaTensor_nElement(state, target), 2,
    "input and target need to have the same number of elements"
  );
  if (mask)
    THArgCheck(
      THCudaTensor_nElement(state, mask) == THCudaTensor_nElement(state, target), 2,
      "mask and target need to have the same number of elements"
    );

  long size = THCudaTensor_nElement(state, input);
  float norm = 0; 
  if (sizeAverage && mask) 
    norm = 1./(THCudaTensor_sumall(state, mask)+EPS);
  else if (sizeAverage)
    norm = 1./size;
  else 
    norm = 1.;

  input = THCudaTensor_newContiguous(state, input);
  target = THCudaTensor_newContiguous(state, target);
  mask = mask ? THCudaTensor_newContiguous(state, mask) : NULL;

  THCudaTensor_resizeAs(state, gradInput, input);

  thrust::device_ptr<float> input_data(THCudaTensor_data(state, input));
  thrust::device_ptr<float> target_data(THCudaTensor_data(state, target));
  thrust::device_ptr<float> mask_data(mask ? THCudaTensor_data(state, mask) : 0);
  thrust::device_ptr<float> gradInput_data(THCudaTensor_data(state, gradInput));
  
  if (mask) 
    thrust::for_each(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      thrust::make_zip_iterator(thrust::make_tuple(input_data,      target_data,      mask_data,      gradInput_data)),
      thrust::make_zip_iterator(thrust::make_tuple(input_data+size, target_data+size, mask_data+size, gradInput_data+size)),
      smoothl1_w_mask_updateGradInput_functor(norm)
    );
  else {
    thrust::transform(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(THCState_getCurrentStream(state)),
#endif
      input_data, input_data+size, target_data, gradInput_data,
      smoothl1_updateGradInput_functor(norm)
    );
  }

  if (mask)
    THCudaTensor_free(state, mask);
  THCudaTensor_free(state, input);
  THCudaTensor_free(state, target);
}

#undef EPS
