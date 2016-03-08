REGIMES = {
    -- start, end,    LR,   WD,    Momentum
    {  1,      7,    5e-4,   1e-4,  0.9},
    {  8,      15,   1e-4,   1e-4,  0.9},
    { 16,      20,   5e-5,   1e-4,  0.9},
    { 21,      50,   1e-5,   1e-4,  0.9}
  }

local function MinMaxPool(kW, kH, dW, dH, padW, padH)
  local layer = nn.Concat(2)
  layer:add(nn.MBSpatialMinPooling(kW, kH, dW, dH, padW, padH))
  layer:add(cudnn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH))
  return layer
end


local model = nn.Sequential() 

-- Convolution Layers
model:add(cudnn.SpatialConvolution(3, 50, 5, 5, 1, 1, 2, 2))
model:add(MinMaxPool(3, 3, 2, 2))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialCrossMapLRN())
model:add(nn.Dropout(0.3))

model:add(cudnn.SpatialConvolution(100, 50, 5, 5, 1, 1, 2, 2))
model:add(MinMaxPool(3, 3, 2, 2))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialCrossMapLRN())
model:add(nn.Dropout(0.45))
  
model:add(cudnn.SpatialConvolution(100, 100, 5, 5, 1, 1, 1, 1))
model:add(MinMaxPool(3, 3, 2, 2))
model:add(cudnn.ReLU())
model:add(nn.Dropout(0.60))

-- Fully Connected Layers   

model:add(cudnn.SpatialConvolution(200, 400, 1, 1, 1, 1))
model:add(cudnn.ReLU())
model:add(nn.Dropout(0.7))
model:add(cudnn.SpatialConvolution(400, 10, 1, 1, 1, 1))
--model:add(cudnn.ReLU())
--model:add(nn.Dropout(0.5))


model:add(nn.Reshape(10))
model:add(nn.SoftMax())

return model
