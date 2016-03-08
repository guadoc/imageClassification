require 'gnuplot'


local trainLogFile = paths.concat(opt.save, 'trainScores.t7')
local valLogFile = paths.concat(opt.save, 'valScores.t7')
if paths.filep(trainLogFile) and opt.lastEpoch > 0 then
    TRAIN_SCORES = torch.load(trainLogFile):narrow(2,1,opt.lastEpoch)
else
    TRAIN_SCORES = torch.Tensor(2,1):fill(opt.nClasses/opt.nTrain)
end
if paths.filep(valLogFile) and opt.lastEpoch > 0 then 
    VAL_SCORES = torch.load(valLogFile):narrow(2,1,opt.lastEpoch)
else
    VAL_SCORES = torch.Tensor(3, 1):fill(opt.nClasses/opt.nVal)
end


function initTrainStats() 
  local stat = {
  loss = 0,
  top1 = 0
  }
  return stat
end

function initValStats() 
  local stat = {
  loss = 0,
  top1 = 0,
  top5 = 0
  }
  return stat
end

function initTestStats() 
  local stat = {
  top1 = 0,
  top5 = 0
  }
  return stat
end

function updateTrainStats(epochStats, bs, loss, preds, labels)
  local top1 = 0
  local _, pred_ind = torch.max(preds, 2) -- descending
  --local _, label_indexes = torch.max(labels, 2) 
  for i=1, preds:size(1) do
    --if labels[i][prediction_indexes[i][1]] == 1 then  
    if pred_ind[i][1] == labels[i] then  
      top1 = top1 + 1
    end
  end  
  epochStats.loss = epochStats.loss + loss
  epochStats.top1 = epochStats.top1 + top1  
  if opt.verbose then 
    print(string.format("Batch: [%d/%d] Time %.3f  DataLoadingTime %.3f  \t batch accuracy: [%.3f %%], \t loss: [%.2f]", bs.batchNumber, bs.nBatch, bs.batchTime, bs.dataLoadingTime, top1*100/preds:size(1), loss))
  end
  
  return epochStats
end

function updateValStats(epochStats, bs, loss, preds, labels)
  local top1 = 0
  local top5 = 0
  local _, pred_ind = preds:sort(2, true) -- descending
  --local _, label_indexes = torch.max(labels, 2) 
  for i=1, preds:size(1) do
    --if labels[i][prediction_indexes[i][1]] == 1 then  
    local g = labels[i]
    local inds = pred_ind[i]
    if inds[1] == g then  
      top1 = top1 + 1
      top5 = top5 + 1
    elseif inds[2] == g or inds[3] == g or inds[4] == g or inds[5] == g  then
      top5 = top5 + 1
    end
  end  
  epochStats.loss = epochStats.loss + loss
  epochStats.top1 = epochStats.top1 + top1  
  epochStats.top5 = epochStats.top5 + top5  
  
  if opt.verbose then 
    print(('Batch: [%d/%d] batch accuracy: [%.2f %%],\t loss: [%.2f]'):format(bs.batchNumber, bs.nBatch, top1*100/preds:size(1), loss/preds:size(1)))
  end
  return epochStats
end

function updateTestStats(epochStats, bs, loss, preds, labels)
  local top1 = 0
  local top5 = 0
  for im = 0, preds:size(1)/opt.testDataAugment -1 do
    im_preds = preds:narrow(1, im*opt.testDataAugment +1, opt.testDataAugment)
    local means = torch.mean(im_preds, 1)   
    local _, pred_sorted = means:sort(2, true)
    
    local g = labels[im*opt.testDataAugment +1]
    if pred_sorted[1][1] == g then 
      top1 = top1 + 1 
      top5 = top5  + 1 
    elseif pred_sorted[1][2] == g or pred_sorted[1][3] == g or pred_sorted[1][4] == g or pred_sorted[1][5] == g then
        top5  = top5  + 1 
    end
  end
  epochStats.top1 = epochStats.top1 + top1
  epochStats.top5 = epochStats.top5 + top5  
  if opt.verbose then 
    print(('Batch: [%d/%d] top1: %.2f ,\t top5: %.2f '):format(bs.batchNumber, bs.nBatch, top1, top5))
  end
  return epochStats
end



function saveTrainStats(epochStats, epoch, time) 
  local avgLoss = epochStats.loss / (opt.nTrain * opt.dataAugment)
  local percenttop1 = epochStats.top1* 100 / (opt.nTrain * opt.dataAugment)
  local y = torch.Tensor(TRAIN_SCORES:size(1), TRAIN_SCORES:size(2)+1):fill(0)
  y:narrow(2,1,TRAIN_SCORES:size(2)):copy(TRAIN_SCORES)
  y[1][TRAIN_SCORES:size(2)+1] = avgLoss
  y[2][TRAIN_SCORES:size(2)+1] = percenttop1
  TRAIN_SCORES  = y    
  print(string.format("TRAIN [%d]: time:%.2d:%.2d:%.2d \t accuracy: top1[%.2f %%],\t loss: [%.4f]", epoch, time/(60*60), time/60%60, time%60, percenttop1, avgLoss)) -- 8 hours  
  torch.save(trainLogFile, TRAIN_SCORES)
end


function saveValStats(epochStats, epoch, time) 
  local avgLoss = epochStats.loss / opt.nVal
  local percenttop1 = epochStats.top1* 100 / opt.nVal
  local percenttop5 = epochStats.top5* 100 / opt.nVal
    
  local y = torch.Tensor(VAL_SCORES:size(1), VAL_SCORES:size(2)+1):fill(0)
  y:narrow(2,1,VAL_SCORES:size(2)):copy(VAL_SCORES)
  y[1][VAL_SCORES:size(2)+1] = avgLoss
  y[2][VAL_SCORES:size(2)+1] = percenttop1
  y[3][VAL_SCORES:size(2)+1] = percenttop5--percenttop5
  VAL_SCORES  = y 
  
  print(string.format("VALID [%d]: time:%.2d:%.2d:%.2d \t accuracy: top1[%.2f %%], top5[%.2f %%], \t loss: [%.3f]", epoch, time/(60*60), time/60%60, time%60, percenttop1, percenttop5, avgLoss)) 
  torch.save(valLogFile, VAL_SCORES)
end



function plotPerformances()
  gnuplot.figure(1)
  gnuplot.plot({'train(top1)',TRAIN_SCORES[2],'~'},{'val(top1)',VAL_SCORES[2],'~'},{'val(top5)',VAL_SCORES[3],'~'})
  gnuplot.grid(true)
  gnuplot.figure(2)
  gnuplot.plot({'train(loss)',TRAIN_SCORES[1],'~'},{'val(loss)',VAL_SCORES[1],'~'})
  gnuplot.grid(true)
end





