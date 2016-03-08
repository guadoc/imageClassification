REGIMES = {
    -- start, end,    LR,   WD,    Momentum
    {  1,      130,   1e-1,   1e-3,  0.9},
    { 131,     160,   1e-2,   1e-4,  0.9},
    { 31,      50,   1e-3,   1e-4,  0.9},
    { 51,      70,   1e-4,   1e-4,  0.9}, 
    { 71,      100,  1e-5,   1e-4,  0.9}    
  }

local function MinMaxPool(kW, kH, dW, dH, padW, padH)
  local layer = nn.Concat(2)
  layer:add(nn.MBSpatialMinPooling(kW, kH, dW, dH, padW, padH))
  layer:add(cudnn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH))
  return layer
end


local model = nn.Sequential() 

-- Convolution Layers
model:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 1, 1))
model:add(MinMaxPool(3, 3, 2, 2))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialCrossMapLRN())


model:add(cudnn.SpatialConvolution(64, 32, 5, 5, 1, 1, 1, 1))
model:add(MinMaxPool(3, 3, 2, 2))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialCrossMapLRN())

  
model:add(cudnn.SpatialConvolution(64, 32, 5, 5, 1, 1, 1, 1))
model:add(MinMaxPool(3, 3, 2, 2))
model:add(cudnn.ReLU())
-- Fully Connected Layers   

model:add(cudnn.SpatialConvolution(64, 10, 1, 1, 1, 1))
--model:add(cudnn.ReLU())
--model:add(nn.Dropout(0.5))


model:add(nn.Reshape(10))
model:add(nn.SoftMax())

return model
