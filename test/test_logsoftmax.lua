require 'nn'
require 'objrcg'

torch.manualSeed(0)

local batch_size = 10
local n_classes  = 5
local model1 = nn.LogSoftMax()
local model2 = objrcg.LogSoftMax()

local input = torch.rand(batch_size, n_classes)
print(input)

local output1 = model1:forward(input)
print(output1)

model2:cuda()
local output2 = model2:forward(input:cuda())
print(output2) 

