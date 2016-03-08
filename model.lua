print('[MODEL]-----------------------------------------------------------')

require 'nn'
require 'cunn'
require 'cudnn'

function paramsForEpoch_g(epoch, regimes)
  local optimState = {}
  optimState.dampening = 0.0
  optimState.learningRateDecay = 0.0
  optimState.learningRate=opt.LR
  optimState.weightDecay=opt.weightDecay
  optimState.momentum = opt.momentum
  for _, row in ipairs(regimes) do
    if epoch >= row[1] and epoch <= row[2] then      
      optimState.learningRate= row[3]
      optimState.weightDecay= row[4]
      optimState.momentum = row[5]
      if  epoch == row[1] then 
        print('(*) New regime : LR ('.. optimState.learningRate..'), WD('..optimState.weightDecay..'), M('..optimState.momentum..')')
      end
      return optimState
    end
  end  
  return optimState
end

function sanitize_g(net)
  local list = net:listModules()
  for _,val in ipairs(list) do
        for name,field in pairs(val) do
           if torch.type(field) == 'cdata' then val[name] = nil end
           if (name == 'output' or name == 'gradInput') then
              val[name] = field.new()
           end
        end
  end
end


local modelFile = paths.concat(opt.modelPath, opt.netType .. '.lua')
assert(paths.filep(modelFile), 'Model not found: ' .. modelFile)
MODEL = paths.dofile(modelFile) --important to get global variable REGIMES
if opt.lastEpoch > 0 then
  local modelFile = paths.concat(opt.save, opt.netType..'_'..opt.lastEpoch..'.t7')
  assert(paths.filep(modelFile), 'File not found: ' .. modelFile)
  print('-----> Loading model from file: ' .. modelFile)
  MODEL = torch.load(modelFile)
end
-- 2. Create Criterion
--CRITERION = nn.MSECriterion()
CRITERION = nn.ClassNLLCriterion()

if opt.nGPU > 0 then
    MODEL:cuda() 
    CRITERION:cuda()
end

if opt.verbose then 
  print('=> Model')
  print(MODEL)
  print('=> Criterion')
  print(CRITERION)
end
print('####### Model ' .. opt.netType ..  ' contains '.. MODEL:getParameters():nElement() .. ' parameters')
collectgarbage()
