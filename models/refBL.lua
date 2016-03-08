REGIMES = {
    -- start, end,    LR,   WD,    Momentum
    {  1,      70,   1e-1,   1e-3,  0.9},
    { 71,      90,   1e-2,   1e-3,  0.9},
    { 41,      50,   1e-4,   1e-3,  0.9},
    { 51,      70,   1e-5,   1e-3,  0.9}, 
    { 71,      100,  1e-5,   1e-3,  0.9}    
  }

local model = nn.Sequential() 

-- Convolution Layers
model:add(cudnn.SpatialConvolution(3, 64, 5, 5, 1, 1, 1, 1))
model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialCrossMapLRN())

model:add(cudnn.SpatialConvolution(64, 64, 5, 5, 1, 1, 1, 1))
model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
model:add(cudnn.ReLU())
model:add(cudnn.SpatialCrossMapLRN())
--model:add(nn.Dropout(0.25))
  
model:add(cudnn.SpatialConvolution(64, 64, 5, 5, 1, 1, 1, 1))
model:add(cudnn.SpatialMaxPooling(3, 3, 2, 2))
model:add(cudnn.ReLU())


-- Fully Connected Layers   
--model:add(cudnn.SpatialConvolution(64, 10, 1, 1, 1, 1))
model:add(nn.Reshape(64))
model:add(nn.Linear(64, 10))
--model:add(cudnn.ReLU())
--model:add(nn.Dropout(0.5))


--odel:add(nn.Reshape(10))
model:add(nn.SoftMax())

return model
