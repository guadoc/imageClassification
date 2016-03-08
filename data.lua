
local ffi = require 'ffi'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
-------------------------------------------------------------------------------
print('[DATASETS]-----------------------------------------------------------')
do -- start K datathreads (donkeys)
   if opt.nDonkeys > 0 then
      local options = opt -- make an upvalue to serialize over to donkey threads
      donkeys = Threads(
         opt.nDonkeys,
         function()
            require 'torch'
         end,
         function(idx)
            opt = options -- pass to all donkeys via upvalue
            tid = idx
            local seed = opt.manualSeed + idx
            torch.manualSeed(seed)
            if opt.verbose then print(string.format('Starting donkey with id: %d seed: %d', tid, seed)) end
            paths.dofile('donkey.lua')
         end
      );
   else -- single threaded data loading. useful for debugging
      paths.dofile('donkey.lua')
      donkeys = {}
      function donkeys:addjob(f1, f2) f2(f1()) end
      function donkeys:synchronize() end
   end
end

local nClasses = nil
local classes = nil
local ntrain = 0
local nVal = 0
local nTest = 0
local nTrain = 0

if opt.train then 
  donkeys:addjob(function() return trainLoader.classes end, function(c) classes = c end)
  donkeys:synchronize()
  nClasses = #classes
  assert(nClasses, "Failed to get nClasses")
  assert(nClasses == opt.nClasses,
         "nClasses is reported different in the data loader, and in the commandline options")
  print('nClasses: ', nClasses)
  torch.save(paths.concat(opt.save, 'classes.t7'), classes)
  donkeys:addjob(function() return trainLoader:size() end, function(c) nTrain = c end)
  donkeys:synchronize()
  assert(nTrain> 0 and nTrain == opt.nTrain * opt.dataAugment, "Failed to get nTrain")
  print('nTrain: ', nTrain)

  donkeys:addjob(function() return valLoader:size() end, function(c) nVal = c end)
  donkeys:synchronize()
  assert(nVal> 0 and nVal == opt.nVal, "Failed to get nVal")
  print('nVal: ', nVal)
else
  donkeys:addjob(function() return testLoader:size() end, function(c) nTest = c end)
  donkeys:synchronize()
  assert(nTest> 0 and nTest == opt.nTest, "Failed to get nTest")
  print('nTest: ', nTest)
end

print('####### Datasets OK')
