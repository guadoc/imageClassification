--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--require 'image'
paths.dofile('dataset.lua')
--paths.dofile('util.lua')

local valList = {paths.concat(opt.data , 'cifar-10-batches-t7/test_batch.t7')}
local trainList = {}
for i = 0,4 do
  trainList[#trainList + 1] = paths.concat(opt.data , 'cifar-10-batches-t7/data_batch_' .. (i+1) .. '.t7')
end

if opt.train then
  print('train Set')
  trainLoader = dataLoader(trainList, opt.nTrain, opt.dataAugment, opt.preProcess)
  print('valisation Set')
  valLoader   = dataLoader(valList, opt.nVal, 0, opt.preProcess)
else
  print('test Set')
  testLoader   = dataLoader(valList, opt.nTest, 0, 0)
end

collectgarbage()

