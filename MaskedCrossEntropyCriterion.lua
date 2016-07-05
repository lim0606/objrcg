local MaskedCrossEntropyCriterion, Criterion = torch.class('objrcg.MaskedCrossEntropyCriterion', 'nn.Criterion')

function MaskedCrossEntropyCriterion:__init(weights)
   Criterion.__init(self)
   self.lsm = objrcg.LogSoftMax()
   self.nll = objrcg.MaskedClassNLLCriterion(weights)
end

function MaskedCrossEntropyCriterion:forward(input, target, mask)
   return self:updateOutput(input, target, mask)
end

function MaskedCrossEntropyCriterion:backward(input, target, mask)
   return self:updateGradInput(input, target, mask)
end

function MaskedCrossEntropyCriterion:updateOutput(input, target, mask)
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

function MaskedCrossEntropyCriterion:updateGradInput(input, target, mask)
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

function MaskedCrossEntropyCriterion:__call__(input, target, mask)
   self.output = self:forward(input, target, mask)
   self.gradInput = self:backward(input, target, mask)
   return self.output, self.gradInput
end
