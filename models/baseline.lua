
REGIMES = {
    -- start, end,    LR,   WD,    Momentum
    --{ 1,       35,   5e-2,   1e-4,  0.9},
    { 1,       40,   1e-2,   1e-4,  0.9},
    { 41,      45,   5e-3,   1e-4,  0.9},
    { 46,      50,   1e-3,   1e-4,  0.9}, 
    { 51,      70,   1e-4,   1e-4,  0.9} 
  }

local model = nn.Sequential() 
-- Convolution Layers
model:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2))
model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
model:add(cudnn.ReLU())

model:add(cudnn.SpatialConvolution(32, 32, 5, 5, 1, 1, 2, 2))
model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
model:add(cudnn.ReLU())
--model:add(nn.Dropout(0.25))
  
model:add(cudnn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
model:add(cudnn.ReLU())


-- Fully Connected Layers   
model:add(cudnn.SpatialConvolution(64, 64, 4, 4, 1, 1))
model:add(cudnn.ReLU())
--model:add(nn.Dropout(0.5))

model:add(cudnn.SpatialConvolution(64, 10, 1, 1, 1, 1))
--model:add(cudnn.ReLU())

model:add(nn.Reshape(10))
model:add(nn.SoftMax())

return model
