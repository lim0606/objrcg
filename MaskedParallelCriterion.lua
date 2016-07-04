local MaskedParallelCriterion, parent = torch.class('objrcg.MaskedParallelCriterion', 'nn.Criterion')

function MaskedParallelCriterion:__init(repeatTarget)
   parent.__init(self)
   self.criterions = {}
   self.weights = {}
   self.gradInput = {}
   self.repeatTarget = repeatTarget
end

function MaskedParallelCriterion:add(criterion, weight)
   assert(criterion, 'no criterion provided')
   weight = weight or 1
   table.insert(self.criterions, criterion)
   table.insert(self.weights, weight)
   return self
end

function MaskedParallelCriterion:forward(input, target, mask)
   return self:updateOutput(input, target, mask)
end

function MaskedParallelCriterion:backward(input, target, mask)
   return self:updateGradInput(input, target, mask)
end

function MaskedParallelCriterion:updateOutput(input, target, mask)
   self.output = 0
   for i,criterion in ipairs(self.criterions) do
      local target = self.repeatTarget and target or target[i]
      local mask   = self.repeatTarget and mask or mask[i]
      self.output = self.output + self.weights[i]*criterion:updateOutput(input[i], target, mask)
   end
   return self.output
end

function MaskedParallelCriterion:updateGradInput(input, target, mask)
   self.gradInput = nn.utils.recursiveResizeAs(self.gradInput, input)
   nn.utils.recursiveFill(self.gradInput, 0)
   for i,criterion in ipairs(self.criterions) do
      local target = self.repeatTarget and target or target[i]
      local mask   = self.repeatTarget and mask or mask[i]
      nn.utils.recursiveAdd(self.gradInput[i], self.weights[i], criterion:updateGradInput(input[i], target, mask))
   end
   return self.gradInput
end

function MaskedParallelCriterion:type(type, tensorCache)
   self.gradInput = {}
   return parent.type(self, type, tensorCache)
end

function MaskedParallelCriterion:__call__(input, target, mask)
   self.output = self:forward(input, target, mask)
   self.gradInput = self:backward(input, target, mask)
   return self.output, self.gradInput
end
