--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

require 'torch'
require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')
local opts = paths.dofile('config.lua')
opt = opts.parse(arg)

opt.train = false

paths.dofile('util.lua')
paths.dofile('model.lua')
paths.dofile('data.lua')
paths.dofile('controle.lua')


if opt.train then 
  paths.dofile('train.lua')
  paths.dofile('valid.lua')
  for epoch=opt.lastEpoch + 1, opt.nEpochs do   
    train(epoch)
    valid(epoch)  
    plotPerformances()
    collectgarbage()
  end
else
  paths.dofile('test.lua')
  test()
end




