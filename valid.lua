

local batchNumber
local epochStats
local nBatch
local timer = torch.Timer()

function valid(epoch)

  batchNumber = 0
  cutorch.synchronize()
  timer:reset()
  local nVal = opt.nVal
  nBatch = math.ceil(opt.nVal / opt.batchSize) 
  MODEL:evaluate()
  epochStats = initValStats()
  
  for batch = 0, nBatch -1 do -- nTest is set in 1_data.lua
    local indexStart = batch * opt.batchSize + 1
    local indexEnd = math.min(indexStart + opt.batchSize - 1, nVal)
    donkeys:addjob(
         -- work to be done by donkey thread
      function()
         local inputs, labels = valLoader:get(indexStart, indexEnd)
        return inputs, labels
      end,
      -- callback that is run in the main thread once the work is done
      testBatch
    )
    batchNumber = batch + 1
    xlua.progress(batch, nBatch)
  end

  donkeys:synchronize()
  cutorch.synchronize()

  saveValStats(epochStats, epoch, timer:time().real)
  collectgarbage()

end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function testBatch(inputsCPU, labelsCPU)
  local batchStat = {batchNumber = batchNumber, nBatch = nBatch}
  
  inputs:resize(inputsCPU:size()):copy(inputsCPU)
  labels:resize(labelsCPU:size()):copy(labelsCPU)

  local outputs = MODEL:forward(inputs)
  local err = CRITERION:forward(outputs, labels)
  
  cutorch.synchronize()  
  epochStats = updateValStats(epochStats, batchStat, err, outputs:float(), labelsCPU)   
end
