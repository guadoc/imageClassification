--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
   local defaultDir = './'

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Imagenet Training script')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-verbose',            false, 'display more message if true')
    cmd:option('-cache',              paths.concat(defaultDir, 'expe/'), 'subdirectory in which to save/log experiments')
    cmd:option('-data',               paths.concat(defaultDir, 'data/'),  'Home of ImageNet dataset')
    cmd:option('-modelPath',          './models/',   'Home of models')    
    cmd:option('-manualSeed',         2, 'Manually set RNG seed')
    cmd:option('-nGPU',               1, 'Number of GPUs to use by default')    
    cmd:option('-nDonkeys',           1, 'number of donkeys to initialize (data loading threads)')
    ------------- Data options ------------------------
    cmd:option('-nTrain',         50000, 'train set size')
    cmd:option('-nVal',           10000, 'validation set size')
    cmd:option('-nTest',          10000, 'test set size')
    
    cmd:option('-channelDim',         3, '1 | 3 number of channel')
    cmd:option('-imageSize',         32,    'Smallest side of the resized image')
    cmd:option('-inputSize',         24,    'Height and Width of image crop to be used as input layer')
    cmd:option('-nClasses',          10, 'number of classes in the dataset')
    
    cmd:option('-dataAugment',       36, 'number of classes in the dataset')
    cmd:option('-testDataAugment',   36, 'number of classes in the dataset')
    cmd:option('-preProcess',        true, 'number of classes in the dataset')
    ------------- Training options --------------------
    cmd:option('-nEpochs',           100,    'Number of total epochs to run')
    cmd:option('-batchSize',         100,   'mini-batch size (1 = pure stochastic)')    
    cmd:option('-lastEpoch',         0,     'Manual epoch number (useful on restarts)')
    ---------- Optimization options ----------------------
    
    cmd:option('-LR',              0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     5e-4, 'weight decay')
    ---------- Model options ----------------------------------
    cmd:option('-netType',         'optim5', 'Options: alexnet | overfeat | alexnetowtbn | vgg | googlenet')
    --cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:text()

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    opt.save = paths.concat(opt.cache, opt.netType)      
    os.execute('mkdir -p ' .. opt.save)
    cmd:log(opt.save .. '/Log.txt', opt)
    
    if opt.verbose then 
      print('Saving everything to: ' .. opt.save)
    else
      cmd.silentio = false
    end
    
    if opt.nGPU > 0 then 
      cutorch.setDevice(opt.nGPU)
      cutorch.manualSeed(opt.manualSeed)      
    end
    

    return opt
end

return M
