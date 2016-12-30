require 'torch'
require 'optim'
require 'xlua'
require 'image'
dofile 'etc.lua'

params, gradParams = model:getParameters()
optimState = {
    learningRate = lr,
    beta1 = b1,
}
optimMethod = optim.adam
tot_error = 0
cnt_error = 0
tot_iter = 0

function train(trainData, trainLabel)
    local time = sys.clock()
    
    tot_error = 0
    cnt_error = 0

    model:training()
    shuffle = torch.randperm(trainDataSz)
    
    local inputs = torch.CudaTensor(batchSz,inputDim,inputSz,inputSz)
    local targets = torch.CudaTensor(batchSz,inputDim,outputSz,outputSz)

    for t = 1,trainDataSz,batchSz do
        
        if t+batchSz-1 > trainDataSz then
            inputs = torch.CudaTensor(trainDataSz-t+1,inputDim,inputSz,inputSz)
            targets = torch.CudaTensor(trainDataSz-t+1,inputDim,outputSz,outputSz)
            curBatchDim = trainDataSz-t+1
        else
            curBatchDim = batchSz
        end

        for i = t,math.min(t+batchSz-1,trainDataSz) do
            
            local input_name = trainData[shuffle[i]]
            local target_name = trainLabel[shuffle[i]]
            local input_ = image.load(input_name)
            local target_ = image.load(target_name)
            
            local inputH = input_:size()[2]
            local inputW = input_:size()[3]
            local targetH = target_:size()[2]
            local targetW = target_:size()[3]
            
            local x_rand = torch.random(1,inputW-inputSz+1)
            local y_rand = torch.random(1,inputH-inputSz+1)

            local input = input_[{{},{y_rand,y_rand+inputSz-1},{x_rand,x_rand+inputSz-1}}]
            local target = target_[{{},{(y_rand-1)*poolFactor+1, (y_rand-1)*poolFactor+outputSz},{(x_rand-1)*poolFactor+1,(x_rand-1)*poolFactor+outputSz}}]
            
            input = torch.reshape(input,inputDim,inputSz,inputSz)
            target = torch.reshape(target,inputDim,outputSz,outputSz)
            
            --[===[
            if t==1 and i<t+20 then
                image.save(tostring(i) .. ".jpg",input)
                image.save(tostring(i) .. "_.jpg",target)
            end
            --]===]
                        
            inputs[i-t+1]:copy(input)
            targets[i-t+1]:copy(target)
        end
        
            local feval = function(x)
            if x ~= params then
                params:copy(x)
            end

            gradParams:zero()
            output = model:forward(inputs)
            
            err = criterion:forward(output,targets)/(outputSz*outputSz*curBatchDim)
            dfdo = criterion:backward(output,targets)/(outputSz*outputSz*curBatchDim)

            model:backward(inputs,dfdo)
            tot_error = tot_error + err
            cnt_error = cnt_error + 1
            tot_iter = tot_iter + 1
            
            gradParams:div(curBatchDim)
            return err,gradParams
            end

         optimMethod(feval, params, optimState)
        
        if tot_iter % 1000 == 0 then
            print("iter: " .. tot_iter .. "/" .. iterLimit .. " batch: " ..  t .. "/" .. trainDataSz .. " loss: " .. tot_error/cnt_error)
        end

    end
   
    if epoch == epochNum then
        local filename = paths.concat(save_dir, modelName)
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('==> saving model to '..filename)
        torch.save(filename, model)
    end
end


