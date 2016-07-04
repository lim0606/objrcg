require 'nn'
require 'objrcg'

torch.manualSeed(0)

local batch_size = 10
local n_classes = 5
local fh = 3
local fw = 4

local model = objrcg.LogSoftMax():cuda()
local preinput = torch.rand(batch_size, n_classes, fh, fw)
local input = model:forward(preinput:cuda()):float()
print(input:size())
--print(input)

local target = torch.zeros(batch_size, fh, fw):random(1, n_classes)
print(target:size())
--print(target)

local mask = target:clone():fill(1)
print(mask:size())

local criterion1 = nn.SpatialClassNLLCriterion():cuda()
local loss1 = criterion1:forward(input:cuda(), target:cuda())
print(loss1)
 
local criterion2 = objrcg.MaskedSpatialClassNLLCriterion():cuda()
local loss2a = criterion2:forward(input:cuda(), target:cuda())
print(loss2a)

local loss2b = criterion2:forward(input:cuda(), target:cuda(), mask:fill(1):cuda())
print(loss2b)

local loss2c = criterion2:forward(input:cuda(), target:cuda(), mask:fill(0):cuda())
print(loss2c)

local loss2d = criterion2:forward(input:cuda(), target:cuda(), mask:random(0, 1):cuda())
print(loss2d)

local gradInput1 = criterion1:backward(input:cuda(), target:cuda()):float()
print(gradInput1:size())
local gradInput2a = criterion2:backward(input:cuda(), target:cuda()):float()
print(gradInput2a:size())
local gradInput2b = criterion2:backward(input:cuda(), target:cuda(), mask:fill(1):cuda()):float()
print(gradInput2b:size())
local gradInput2c = criterion2:backward(input:cuda(), target:cuda(), mask:fill(0):cuda()):float()
print(gradInput2c:size())

local diffa = (gradInput1-gradInput2a):view(-1):sum()
print(diffa)
local diffb = (gradInput1-gradInput2b):view(-1):sum()
print(diffb)
local diffc = gradInput2c:view(-1):sum()
print(diffc)
