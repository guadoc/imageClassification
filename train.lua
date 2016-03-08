
require 'optim'

--[[
if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end
]]


local batchNumber
local nBatch
local epochStats
local optimState
-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train(epoch)
  local tm = torch.Timer()
  local nTrain = opt.nTrain * opt.dataAugment
  batchNumber = 0
  cutorch.synchronize()
  optimState = paramsForEpoch_g(epoch, REGIMES)
  nBatch = math.ceil(nTrain / opt.batchSize)  
  --donkeys:addjob(function() trainLoader:shuffle() end, function() end)  
  MODEL:training()
  epochStats = initTrainStats()
  
  for batch=0 , nBatch -1 do
    local indexStart = batch * opt.batchSize + 1
    local indexEnd = math.min(indexStart + opt.batchSize - 1, nTrain)
    donkeys:addjob(         
      function()   -- the job callback (runs in data-worker thread)     
        --local inputs, labels = trainLoader:sample(opt.batchSize)
        local inputs, labels = trainLoader:get(indexStart, indexEnd)
        return inputs, labels
      end,      
      trainBatch  -- the end callback (runs in the main thread)
    )
    batchNumber = batch + 1
    xlua.progress(batch, nBatch)
  end

  donkeys:synchronize()
  cutorch.synchronize()
  
  saveTrainStats(epochStats, epoch, tm:time().real)  
  sanitize_g(MODEL)
  saveDataParallel(paths.concat(opt.save, opt.netType..'_'..epoch..'.t7'), MODEL) 
  torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
  collectgarbage()
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local batchTimer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = MODEL:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
  local batchStat = {dataLoadingTime = dataTimer:time().real, batchNumber = batchNumber, nBatch = nBatch}
  
  cutorch.synchronize()
  collectgarbage()  
  batchTimer:reset()
   -- transfer over to GPU
  inputs:resize(inputsCPU:size()):copy(inputsCPU)
  labels:resize(labelsCPU:size()):copy(labelsCPU)
  local err, outputs
  feval = function(x)
    MODEL:zeroGradParameters()
    outputs = MODEL:forward(inputs)
    err = CRITERION:forward(outputs, labels)
    local gradOutputs = CRITERION:backward(outputs, labels)
    MODEL:backward(inputs, gradOutputs)
    return err, gradParameters
  end
  optim.sgd(feval, parameters, optimState)
   -- DataParallelTable's syncParameters
  MODEL:apply(function(m) if m.syncParameters then m:syncParameters() end end)
  cutorch.synchronize()  
  batchStat.batchTime = batchTimer:time().real
  epochStats = updateTrainStats(epochStats, batchStat, err, outputs:float(), labelsCPU)
  dataTimer:reset()  
end
