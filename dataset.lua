require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

--[[local ffi = require 'ffi'
local class = require('pl.class')
local dir = require 'pl.dir'
local tablex = require 'pl.tablex'
local argcheck = require 'argcheck'
require 'sys'
require 'xlua'
]]

paths.dofile('lib/data_augmentation.lua')
paths.dofile('lib/preprocessing.lua')
require 'image'

local dataset = torch.class('dataLoader')

function dataset:__init(dataFileList, size, augmentData, preProcess)
  if not paths.dirp(paths.concat(opt.data , 'cifar-10-batches-t7')) then
     local www = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
     local tar = paths.basename(www)
     os.execute('cd '..opt.data ..' ; wget ' .. www .. '; '.. 'tar xvf ' .. tar ..'; cd ..')
  end
  
  self.inputs = torch.Tensor(size, 3072)
  self.labels = torch.Tensor(size)
  self.size_ = size 
  
  for i = 1, #dataFileList do
     local subset = torch.load(paths.concat(dataFileList[i]), 'ascii')
     self.inputs[{ {(i-1)*10000+1, i*10000} }] = subset.data:t()
     self.labels[{ {(i-1)*10000+1, i*10000} }] = subset.labels
  end
  self.labels = self.labels + 1
  
  self.inputs = self.inputs[{ {1,size} }]
  self.labels = self.labels[{ {1,size} }]
  self.inputs = self.inputs:reshape(size, opt.channelDim, opt.imageSize, opt.imageSize):div(255)
  
  --[[
  local Label = torch.zeros(size, opt.nClasses)
  for i=1, self.size do
    Label[i][self.labels[i] ] = 1
  end
  self.labels = Label 
  ]]


  if augmentData > 1 then
    print('----->Data Augmentation')
    self.inputs, self.labels, self.size_ = data_augmentation(self.inputs, self.labels)
    opt.nTrain = opt.nTrain * opt.dataAugment
    print('#data augmented')
  elseif opt.train then 
    local inputs = torch.Tensor(self.size_, opt.channelDim, opt.inputSize, opt.inputSize)
    for i=1,self.size_ do
      inputs[i] = image.scale(self.inputs[i], opt.inputSize, opt.inputSize)
    end    
    self.inputs = inputs
  end
  
  
  if preProcess then
    print('----->Data preprocessing')
    local params = nil
    local paramsFile = paths.concat(opt.data, 'preprocessing_params.bin' )
    if not paths.filep(paramsFile) then 
      local params = preprocessing(self.inputs)
      torch.save(paramsFile, params)
      print("#data preprocessed")
      print('#Pre-process params saved in :'..paramsFile)
    else      
      self.params = torch.load(paramsFile)
      print('#Pre-process params loaded from :'..paramsFile)
      if opt.train then
        preprocessing(self.inputs, self.params)           
        print("#data preprocessed")
      end      
    end    
  end
  
  self.classes = {1, 2, 3, 4 , 5, 6, 7, 9, 9, 10}  
end

-- size(), size(class)
function dataset:size()
  return self.size_
end

-- sampler, samples from the training set.
function dataset:sample(quantity)
  local data = nil
  local labels = nil
  return data, labels
end

function dataset:get(i1, i2)  
  local data = self.inputs:narrow(1, i1, i2-i1+1)
  local labels = self.labels:narrow(1, i1, i2-i1+1)
  return data, labels
end

function dataset:getImage(index)  
  local imageData = self.inputs[index]
  local label = self.labels[index]
  return imageData, label
end

function dataset:shuffle()
  local order = torch.LongTensor():randperm(self.size)
  self.inputs = self.inputs:index(1, order)
  self.labels = self.labels:index(1, order)
end


function dataset:getTest(i1, i2)
  local data = self.inputs:narrow(1, i1, i2-i1+1)
  local labels = self.labels:narrow(1, i1, i2-i1+1)
  local inputs
  local size = data:size(1)
  if opt.testDataAugment > 1 then
    inputs, labels, size = data_augmentation(data, labels)
  else 
    inputs = torch.Tensor(size, opt.channelDim, opt.inputSize, opt.inputSize)
    for i=1, size do
      inputs[i] = image.scale(data[i], opt.inputSize, opt.inputSize)
    end     
  end 
  if opt.preProcess then
    preprocessing(inputs, self.params)
  end
  return inputs, labels
end


return dataset
