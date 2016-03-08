
REGIMES = {
    -- start, end,    LR,   WD,    Momentum
    {  1,      40,   5e-1,   1e-4,  0.9},
    { 41,      48,   1e-1,   1e-4,  0.9},
    { 49,      55,   5e-2,   1e-4,  0.9},
    { 56,      60,   1e-2,   1e-4,  0.9}, 
    { 61,      65,   1e-3,   1e-4,  0.9},
    { 66,      90,   1e-4,   1e-4,  0.9}
  }

local model = nn.Sequential() 

-- Convolution Layers
model:add(cudnn.SpatialConvolution(3, 350, 5, 5, 1, 1, 2, 2))
model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
model:add(cudnn.ReLU())

model:add(cudnn.SpatialConvolution(350, 350, 5, 5, 1, 1, 2, 2))
model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
model:add(cudnn.ReLU())
--model:add(nn.Dropout(0.25))
  
model:add(cudnn.SpatialConvolution(350, 550, 5, 5, 1, 1, 2, 2))
model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2, 1, 1))
model:add(cudnn.ReLU())
--model:add(nn.Dropout(0.5))


-- Fully Connected Layers   
model:add(cudnn.SpatialConvolution(550, 900, 4, 4, 1, 1))
model:add(cudnn.ReLU())
--model:add(nn.Dropout(0.5))

model:add(cudnn.SpatialConvolution(900, 10, 1, 1, 1, 1))
--model:add(cudnn.ReLU())

model:add(nn.Reshape(10))
model:add(nn.SoftMax())

return model
