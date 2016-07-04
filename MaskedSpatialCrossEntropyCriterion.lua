local MaskedSpatialCrossEntropyCriterion, Criterion = torch.class('objrcg.MaskedSpatialCrossEntropyCriterion', 'nn.Criterion')

function MaskedSpatialCrossEntropyCriterion:__init(weights)
   Criterion.__init(self)
   self.lsm = objrcg.LogSoftMax()
   self.nll = objrcg.MaskedSpatialClassNLLCriterion(weights)
end

function MaskedSpatialCrossEntropyCriterion:forward(input, target, mask)
   return self:updateOutput(input, target, mask)
end

function MaskedSpatialCrossEntropyCriterion:backward(input, target, mask)
   return self:updateGradInput(input, target, mask)
end

function MaskedSpatialCrossEntropyCriterion:updateOutput(input, target, mask)
   --input = input:squeeze()
   --target = type(target) == 'number' and target or target:squeeze()
   assert(torch.type(input) == 'torch.CudaTensor', 'Current implementation only support cuda-version.')
   if (torch.type(input) == 'torch.CudaTensor') then
     self.lsm:cuda()
     self.nll:cuda()
   end
   self.lsm:updateOutput(input)
   self.nll:updateOutput(self.lsm.output, target, mask)
   self.output = self.nll.output
   return self.output
end

function MaskedSpatialCrossEntropyCriterion:updateGradInput(input, target, mask)
   local size = input:size()
   --input = input:squeeze()
   --target = type(target) == 'number' and target or target:squeeze()
   assert(torch.type(input) == 'torch.CudaTensor', 'Current implementation only support cuda-version.')
   if (torch.type(input) == 'torch.CudaTensor') then
     self.lsm:cuda()
     self.nll:cuda()
   end
   self.nll:updateGradInput(self.lsm.output, target)
   self.lsm:updateGradInput(input, self.nll.gradInput, mask)
   self.gradInput:view(self.lsm.gradInput, size)
   return self.gradInput
end

function MaskedSpatialCrossEntropyCriterion:__call__(input, target, mask)
   self.output = self:forward(input, target, mask)
   self.gradInput = self:backward(input, target, mask)
   return self.output, self.gradInput
end
