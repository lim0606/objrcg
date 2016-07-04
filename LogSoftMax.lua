local LogSoftMax = torch.class('objrcg.LogSoftMax', 'nn.Module')
local C = objrcg.C

function LogSoftMax:updateOutput(input)
   --if torch.type(input) == 'torch.CudaTensor' then
     assert(torch.type(input) == 'torch.CudaTensor', 'Current implementation only support cuda-version.')
     C.THNN_CudaLogSoftMax_updateOutput(
        cutorch.getState(),
        input:cdata(),
        self.output:cdata()
     )
   --else 
   --  C.THNN_LogSoftMax_updateOutput(
   --     input:cdata(),
   --     self.output:cdata()
   --  )
   --end

   return self.output
end

function LogSoftMax:updateGradInput(input, gradOutput)
   --if torch.type(input) == 'torch.CudaTensor' then
     assert(torch.type(input) == 'torch.CudaTensor', 'Current implementation only support cuda-version.')
     C.THNN_CudaLogSoftMax_updateGradInput(
        cutorch.getState(),
        input:cdata(),
        gradOutput:cdata(),
        self.gradInput:cdata(),
        self.output:cdata()
     )
   --else
   --  C.THNN_LogSoftMax_updateGradInput(
   --     input:cdata(),
   --     gradOutput:cdata(),
   --     self.gradInput:cdata(),
   --     self.output:cdata()
   --  )
   --end
   return self.gradInput
end
