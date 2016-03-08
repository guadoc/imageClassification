REGIMES = {
    -- start, end,    LR,   WD,    Momentum
    {  1,      7,    5e-1,   1e-4,  0.9},
    {  8,      15,   1e-1,   1e-4,  0.9},
    { 16,      20,   1e-2,   1e-4,  0.9},
    { 21,      50,   1e-3,   1e-4,  0.9}
  }

local function MinMaxPool(kW, kH, dW, dH, padW, padH)
  local layer = nn.Concat(2)
  layer:add(nn.MBSpatialMinPooling(kW, kH, dW, dH, padW, padH))
  layer:add(cudnn.SpatialMaxPooling(kW, kH, dW, dH, padW, padH))
  return layer
end


local model = nn.Sequential() 

-- Convolution Layers
model:add(cudnn.SpatialConvolution(3, 400, 5, 5, 1, 1, 2, 2))
model:add(MinMaxPool(3, 3, 2, 2))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialCrossMapLRN())
model:add(nn.Dropout(0.3))

model:add(cudnn.SpatialConvolution(800, 500, 5, 5, 1, 1, 2, 2))
model:add(MinMaxPool(3, 3, 2, 2))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialCrossMapLRN())
model:add(nn.Dropout(0.45))
  
model:add(cudnn.SpatialConvolution(1000, 600, 5, 5, 1, 1, 1, 1))
model:add(MinMaxPool(3, 3, 2, 2))
model:add(cudnn.ReLU())
model:add(nn.Dropout(0.60))

-- Fully Connected Layers   

model:add(cudnn.SpatialConvolution(1200, 1900, 1, 1, 1, 1))
model:add(cudnn.ReLU())
model:add(nn.Dropout(0.7))
model:add(cudnn.SpatialConvolution(1900, 10, 1, 1, 1, 1))
--model:add(cudnn.ReLU())
--model:add(nn.Dropout(0.5))


model:add(nn.Reshape(10))
model:add(nn.SoftMax())

return model
