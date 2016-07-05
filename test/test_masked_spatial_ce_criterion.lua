require 'nn'
require 'objrcg'

torch.manualSeed(0)

local batch_size = 100
local n_classes = 5
local fh = 23
local fw = 34

local input = torch.rand(batch_size, n_classes, fh, fw)
print(input:size())
--print(input)

local target = torch.zeros(batch_size, fh, fw):random(1, n_classes)
print(target:size())
--print(target)

local mask = target:clone():random(0, 1)
print(mask:size())
--print(mask)

local lsm = objrcg.LogSoftMax():cuda()
local criterion1 = objrcg.MaskedSpatialClassNLLCriterion():cuda()
local output = lsm:forward(input:cuda())
local loss1 = criterion1:forward(output, target:cuda(), mask:cuda())
print(loss1)
 
local criterion2 = objrcg.MaskedSpatialCrossEntropyCriterion():cuda()
local loss2 = criterion2:forward(input:cuda(), target:cuda(), mask:cuda())
print(loss2)

local gradInput1_ = criterion1:backward(output, target:cuda(), mask:cuda()):float()
local gradInput1 = lsm:backward(input:cuda(), gradInput1_:cuda()):float()
print(gradInput1:size())
local gradInput2 = criterion2:backward(input:cuda(), target:cuda(), mask:cuda()):float()
print(gradInput2:size())

local diff = (gradInput1-gradInput2):view(-1):sum()
print(diff)
