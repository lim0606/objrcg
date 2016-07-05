require 'nn'
objrcg = {}
objrcg.C = require 'objrcg.ffi'
require 'objrcg.LogSoftMax'
require 'objrcg.MaskedClassNLLCriterion'
require 'objrcg.MaskedSpatialClassNLLCriterion'
require 'objrcg.MaskedSpatialCrossEntropyCriterion'
require 'objrcg.MaskedSmoothL1Criterion'
require 'objrcg.MaskedParallelCriterion'
return objrcg
