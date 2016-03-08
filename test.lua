require 'image'

local epochStats
local timer = torch.Timer()
local nBatch
local batchNumber


function test(epoch)
  local batchSize = math.max(math.floor(opt.batchSize / opt.testDataAugment), 1)  
  local nTest = opt.nTest
  
  epochStats = initTestStats()
  timer:reset()
  nBatch = math.ceil(nTest / batchSize)
  batchNumber = 0
  
  cutorch.synchronize()
  MODEL:evaluate()
  
  for batch = 0, nBatch-1 do
    local indexStart = batch * batchSize + 1
    local indexEnd = math.min(indexStart + batchSize - 1, nTest)
    donkeys:addjob(
         -- work to be done by donkey thread
      function()   
        local inputs, labels = testLoader:getTest(indexStart, indexEnd) 
        return inputs, labels
      end, 
      -- callback that is run in the main thread once the work is done
      testImage
    )    
    batchNumber = batch + 1
    xlua.progress(batch, nBatch)

  end  
  print(string.format("TEST :  accuracy: top1[%.2f %%],\t top5: [%.4f]", epochStats.top1*100/nTest, epochStats.top5*100 /nTest))
  donkeys:synchronize()
  cutorch.synchronize()
  
  collectgarbage()
end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

function testImage(inputsCPU, labelsCPU)
  local batchStat = {batchNumber = batchNumber, nBatch = nBatch}
  
  inputs:resize(inputsCPU:size()):copy(inputsCPU)
  labels:resize(labelsCPU:size()):copy(labelsCPU)
  
  local outputs = MODEL:forward(inputs)
  local err = CRITERION:forward(outputs, labels)
  
  cutorch.synchronize()
  epochStats = updateTestStats(epochStats, batchStat, err, outputs:float(), labelsCPU)
end
