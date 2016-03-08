REGIMES = {
    -- start, end,    LR,   WD,    Momentum
    {  1,      28,   5e-1,   1e-4,  0.9},
    { 29,      40,   1e-1,   1e-4,  0.9},
    { 41,      45,   5e-2,   1e-4,  0.9},
    { 46,      50,   1e-2,   1e-4,  0.9}, 
    { 51,      70,   1e-3,   1e-4,  0.9} 
  }

local function MinMaxPool(kW, kH, dW, dH, padW, padH)
  local layer = nn.Concat(2)
  layer:add(nn.MBSpatialMinPooling(kW, kH, dW, dH, padW, padH))
  layer:add(cudnn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH))
  return layer
end


local model = nn.Sequential() 

-- Convolution Layers
model:add(cudnn.SpatialConvolution(3, 128, 5, 5, 1, 1, 2, 2))
model:add(MinMaxPool(3, 3, 2, 2))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialCrossMapLRN())
--model:add(nn.Dropout(0.25))

model:add(cudnn.SpatialConvolution(256, 256, 5, 5, 1, 1, 2, 2))
model:add(MinMaxPool(3, 3, 2, 2))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialCrossMapLRN())
model:add(nn.Dropout(0.25))
  
model:add(cudnn.SpatialConvolution(512, 256, 5, 5, 1, 1, 1, 1))
model:add(MinMaxPool(3, 3, 2, 2))
model:add(cudnn.ReLU())
model:add(nn.Dropout(0.50))

-- Fully Connected Layers   

model:add(cudnn.SpatialConvolution(512, 1024, 1, 1, 1, 1))
model:add(cudnn.ReLU())
model:add(nn.Dropout(0.5))
model:add(cudnn.SpatialConvolution(1024, 10, 1, 1, 1, 1))
--model:add(cudnn.ReLU())
--model:add(nn.Dropout(0.5))


model:add(nn.Reshape(10))
model:add(nn.SoftMax())

return model
