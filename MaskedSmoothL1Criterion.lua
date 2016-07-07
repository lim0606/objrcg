local MaskedSmoothL1Criterion, parent = torch.class('objrcg.MaskedSmoothL1Criterion', 'nn.Criterion')
local C = objrcg.C

local function optionalTensor(t)
   return t and t:cdata() or nil
end

function MaskedSmoothL1Criterion:__init(sigma, sizeAverage)
   parent.__init(self)
   self.sigma = sigma or 1 
   if sizeAverage ~= nil then
     self.sizeAverage = sizeAverage
   else
     self.sizeAverage = true
   end
end

function MaskedSmoothL1Criterion:forward(input, target, mask)
   return self:updateOutput(input, target, mask)
end

function MaskedSmoothL1Criterion:backward(input, target, mask)
   return self:updateGradInput(input, target, mask)
end

function MaskedSmoothL1Criterion:updateOutput(input, target, mask)
   self.output_tensor = self.output_tensor or input.new(1)

   --if torch.type(input) == 'torch.CudaTensor' then
     assert(torch.type(input) == 'torch.CudaTensor', 'Current implementation only support cuda-version.')
     C.THNN_CudaMaskedSmoothL1Criterion_updateOutput(
        cutorch.getState(),
        input:cdata(),
        target:cdata(),
        optionalTensor(mask),
        self.output_tensor:cdata(),
        self.sizeAverage, 
        self.sigma
     )
   --else
   --  C.THNN_MaskedSmoothL1Criterion_updateOutput(
   --     input:cdata(),
   --     target:cdata(),
   --     self.output_tensor:cdata(),
   --     self.sizeAverage
   --  )
   --end

   self.output = self.output_tensor[1]
   return self.output
end

function MaskedSmoothL1Criterion:updateGradInput(input, target, mask)
   --if torch.type(input) == 'torch.CudaTensor' then
     assert(torch.type(input) == 'torch.CudaTensor', 'Current implementation only support cuda-version.')
     C.THNN_CudaMaskedSmoothL1Criterion_updateGradInput(
        cutorch.getState(),
        input:cdata(),
        target:cdata(),
        optionalTensor(mask),
        self.gradInput:cdata(),
        self.sizeAverage,
        self.sigma
     )
   --else 
   --  C.THNN_MaskedSmoothL1Criterion_updateGradInput(
   --     input:cdata(),
   --     target:cdata(),
   --     self.gradInput:cdata(),
   --     self.sizeAverage
   --  )
   --end

   return self.gradInput
end

function MaskedSmoothL1Criterion:__call__(input, target, mask)
   self.output = self:forward(input, target, mask)
   self.gradInput = self:backward(input, target, mask)
   return self.output, self.gradInput
end
