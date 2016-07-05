require 'nn'
require 'objrcg'

torch.manualSeed(0)

local batch_size = 10
local n_classes = 5

local model = objrcg.LogSoftMax():cuda()
local preinput = torch.rand(batch_size, n_classes)
local input = model:forward(preinput:cuda()):float()
print(input:size())
--print(input)

local target = torch.zeros(batch_size):random(1, n_classes)
print(target:size())
--print(target)

local mask = target:clone():fill(1)
print(mask:size())

local criterion1 = nn.ClassNLLCriterion():cuda()
local loss1 = criterion1:forward(input:cuda(), target:cuda())
local gradInput1 = criterion1:backward(input:cuda(), target:cuda()):float()
print(loss1)
 
local criterion2 = objrcg.MaskedClassNLLCriterion():cuda()
local loss2a = criterion2:forward(input:cuda(), target:cuda())
local gradInput2a = criterion2:backward(input:cuda(), target:cuda()):float()
print(loss2a)
print(criterion2.total_weight_tensor[1])

local loss2b = criterion2:forward(input:cuda(), target:cuda(), mask:fill(1):cuda())
local gradInput2b = criterion2:backward(input:cuda(), target:cuda(), mask:fill(1):cuda()):float()
print(loss2b)
print(criterion2.total_weight_tensor[1])

local loss2c = criterion2:forward(input:cuda(), target:cuda(), mask:fill(0):cuda())
local gradInput2c = criterion2:backward(input:cuda(), target:cuda(), mask:fill(0):cuda()):float()
print(loss2c)
print(criterion2.total_weight_tensor[1])

local loss2d = criterion2:forward(input:cuda(), target:cuda(), mask:random(0, 1):cuda())
local gradInput2d = criterion2:backward(input:cuda(), target:cuda(), mask:cuda()):float()
print(loss2d)
print(criterion2.total_weight_tensor[1])

print(gradInput1:size())
print(gradInput2a:size())
print(gradInput2b:size())
print(gradInput2c:size())

local diffa = (gradInput1-gradInput2a):view(-1):sum()
print(diffa)
local diffb = (gradInput1-gradInput2b):view(-1):sum()
print(diffb)
local diffc = gradInput2c:view(-1):sum()
print(diffc)
