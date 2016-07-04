require 'nn'
objrcg = {}
objrcg.C = require 'objrcg.ffi'
require 'objrcg.LogSoftMax'
require 'objrcg.MaskedSpatialClassNLLCriterion'
require 'objrcg.MaskedSmoothL1Criterion'
return objrcg
