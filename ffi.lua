local ffi = require 'ffi'

local libpath = package.searchpath('libobjrcg', package.cpath)
if not libpath then return end

require 'cunn'

ffi.cdef[[
//void THNN_LogSoftMax_updateOutput(
//          THNNState *state,            // library's state
//          THTensor *input,             // input tensor
//          THTensor *output);           // [OUT] output tensor
//void THNN_LogSoftMax_updateGradInput(
//          THNNState *state,            // library's state
//          THTensor *input,             // input tensor
//          THTensor *gradOutput,        // gradient w.r.t. module's output
//          THTensor *gradInput,         // [OUT] gradient w.r.t. input
//          THTensor *output);           // module's output

void THNN_CudaLogSoftMax_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *output);
void THNN_CudaLogSoftMax_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *gradOutput,
          THCudaTensor *gradInput,
          THCudaTensor *output);

void THNN_CudaMaskedClassNLLCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *mask,
          THCudaTensor *output,
          bool sizeAverage,
          THCudaTensor *weights,
          THCudaTensor *total_weight);
void THNN_CudaMaskedClassNLLCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *mask,
          THCudaTensor *gradInput,
          bool sizeAverage,
          THCudaTensor *weights,
          THCudaTensor *total_weight);

void THNN_CudaMaskedSpatialClassNLLCriterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *mask,
          THCudaTensor *output,
          bool sizeAverage,
          THCudaTensor *weights,
          THCudaTensor *total_weight);
void THNN_CudaMaskedSpatialClassNLLCriterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *mask,
          THCudaTensor *gradInput,
          bool sizeAverage,
          THCudaTensor *weights,
          THCudaTensor *total_weight);

void THNN_CudaMaskedSmoothL1Criterion_updateOutput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *mask,
          THCudaTensor *output,
          bool sizeAverage, 
          float sigma);
void THNN_CudaMaskedSmoothL1Criterion_updateGradInput(
          THCState *state,
          THCudaTensor *input,
          THCudaTensor *target,
          THCudaTensor *mask,
          THCudaTensor *gradInput,
          bool sizeAverage, 
          float sigma);
]]

return ffi.load(libpath)
