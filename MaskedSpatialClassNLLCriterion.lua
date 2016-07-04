local MaskedSpatialClassNLLCriterion, parent = torch.class('objrcg.MaskedSpatialClassNLLCriterion', 'nn.Criterion')
local C = objrcg.C

local function optionalTensor(t)
   return t and t:cdata() or nil 
end

function MaskedSpatialClassNLLCriterion:__init(weights, sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
       self.sizeAverage = sizeAverage
    else
       self.sizeAverage = true
    end
    if weights then
       assert(weights:dim() == 1, "weights input should be 1-D Tensor")
       self.weights = weights
    end

    self.output_tensor = torch.zeros(1)
    self.total_weight_tensor = torch.ones(1)
    self.target = torch.zeros(1):long()
end

function MaskedSpatialClassNLLCriterion:__len()
   if (self.weights) then
      return #self.weights
   else
      return 0
   end
end

function MaskedSpatialClassNLLCriterion:forward(input, target, mask)
   return self:updateOutput(input, target, mask)
end

function MaskedSpatialClassNLLCriterion:backward(input, target, mask)
   return self:updateGradInput(input, target, mask)
end

function MaskedSpatialClassNLLCriterion:updateOutput(input, target, mask)
   if type(target) == 'number' then
      if input:type() ~= 'torch.CudaTensor' then
         self.target = self.target:long()
      end
      self.target[1] = target
   elseif target:type() == 'torch.CudaTensor' then
      self.target = target
   else
      self.target = target:long()
   end

   --if torch.type(input) == 'torch.CudaTensor' then
     assert(torch.type(input) == 'torch.CudaTensor', 'Current implementation only support cuda-version.')
     C.THNN_CudaMaskedSpatialClassNLLCriterion_updateOutput(
        cutorch.getState(),
        input:cdata(),
        self.target:cdata(),
        optionalTensor(mask),
        self.output_tensor:cdata(),
        self.sizeAverage,
        optionalTensor(self.weights),
        self.total_weight_tensor:cdata()
     )
   --else
   --  C.THNN_MaskedSpatialClassNLLCriterion_updateOutput(
   --     input:cdata(),
   --     self.target:cdata(),
   --     self.output_tensor:cdata(),
   --     self.sizeAverage,
   --     optionalTensor(self.weights),
   --     self.total_weight_tensor:cdata()
   --  )
   --end
   self.output = self.output_tensor[1]
   return self.output, self.total_weight_tensor[1]
end

function MaskedSpatialClassNLLCriterion:updateGradInput(input, target, mask)
   if type(target) == 'number' then
      self.target[1] = target
   elseif target:type() == 'torch.CudaTensor' then
      self.target = target
   else
      self.target = target:long()
   end

   self.gradInput:resizeAs(input):zero()

   --if torch.type(input) == 'torch.CudaTensor' then
     assert(torch.type(input) == 'torch.CudaTensor', 'Current implementation only support cuda-version.')
     C.THNN_CudaMaskedSpatialClassNLLCriterion_updateGradInput(
        cutorch.getState(),
        input:cdata(),
        self.target:cdata(),
        optionalTensor(mask),
        self.gradInput:cdata(),
        self.sizeAverage,
        optionalTensor(self.weights),
        self.total_weight_tensor:cdata()
     )
   --else
   --  C.THNN_MaskedSpatialClassNLLCriterion_updateGradInput(
   --     input:cdata(),
   --     self.target:cdata(),
   --     self.gradInput:cdata(),
   --     self.sizeAverage,
   --     optionalTensor(self.weights),
   --     self.total_weight_tensor:cdata()
   --  )
   --end

   return self.gradInput
end

function MaskedSpatialClassNLLCriterion:__call__(input, target, mask)
   self.output = self:forward(input, target, mask)
   self.gradInput = self:backward(input, target, mask)
   return self.output, self.gradInput
end
