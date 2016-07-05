require 'nn'
require 'objrcg'

torch.manualSeed(0)

local batch_size = 10
local n_classes = 5
local channels = 4
local fh = 3
local fw = 4

local rpn_CE    = objrcg.MaskedSpatialCrossEntropyCriterion()
local rpn_SL1   = objrcg.MaskedSmoothL1Criterion()
local frcnn_CE  = objrcg.MaskedCrossEntropyCriterion()
local frcnn_SL1 = objrcg.MaskedSmoothL1Criterion()
local criterion = objrcg.MaskedParallelCriterion():add(rpn_CE)
                                                  :add(rpn_SL1)
                                                  :add(frcnn_CE)
                                                  :add(frcnn_SL1)
criterion:cuda()

local input = {}
input[1] = torch.rand(batch_size, n_classes, fh, fw):cuda()
input[2] = torch.rand(batch_size, n_classes * channels, fh, fw):cuda()
input[3] = torch.rand(batch_size, n_classes):cuda()
input[4] = torch.rand(batch_size, n_classes * channels):cuda()

local target = {}
target[1] = torch.zeros(batch_size, fh, fw):random(1, n_classes):cuda()
target[2] = torch.rand(batch_size, n_classes * channels, fh, fw):cuda()
target[3] = torch.rand(batch_size):random(1, n_classes):cuda()
target[4] = torch.rand(batch_size, n_classes * channels):cuda()

local mask = {}
mask[1] = torch.zeros(batch_size, fh, fw):random(0, 1):cuda()
mask[2] = torch.zeros(batch_size, n_classes * channels, fh, fw):random(0, 1):cuda()
mask[3] = nil 
mask[4] = torch.zeros(batch_size, n_classes * channels):random(0, 1):cuda()

local loss = criterion:forward(input, target, mask)
print(loss)

local gradInput = criterion:backward(input, target, mask)
print(gradInput)
