require 'torch'
require 'optim'
require 'xlua'
require 'image'
dofile 'etc.lua'

params, gradParams = combine_and_flatten_parameters(GenModel,DisModel)

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

    GenModel:training()
    DisModel:training()
    shuffle_Gen = torch.randperm(trainDataSz)
    shuffle_Dis = torch.randperm(trainDataSz)
    
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
        
        for k = 1,2 do
            if k == 1 then
                train_model = "D"
            else
                train_model = "G"
            end

            for i = t,math.min(t+batchSz-1,trainDataSz) do
                
                if train_model == "D" then
                    input_name = trainData[shuffle_Dis[i]]
                    target_name = trainLabel[shuffle_Dis[i]]
                else
                    input_name = trainData[shuffle_Gen[i]]
                    target_name = trainLabel[shuffle_Gen[i]]
                end

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

                    output_Gen = GenModel:forward(inputs)
                    
                    if train_model == "D" then
                        --DisModel training
                        Dis_input = torch.CudaTensor(2*curBatchDim,inputDim,outputSz,outputSz)
                        Dis_target = torch.CudaTensor(2*curBatchDim,1)
                        Dis_input[{{1,curBatchDim},{}}] = output_Gen
                        Dis_target[{{1,curBatchDim},{}}] = 1 --give 1 to G
                        Dis_input[{{curBatchDim+1,-1},{}}] = targets
                        Dis_target[{{curBatchDim+1,-1},{}}] = 0 --give 0 to R

                        output_Dis = DisModel:forward(Dis_input)
                        GAN_err = bce:forward(output_Dis,Dis_target)
                        GAN_dfdo = bce:backward(output_Dis,Dis_target)
                        DisModel:backward(Dis_input,GAN_dfdo)

                        if tot_iter % 1000 == 0 then
                            dis_result = output_Dis:gt(0.5)
                            confusion = torch.Tensor(2,2):zero()
                            confusion[1][1] = torch.sum(dis_result[{{curBatchDim+1,-1}}]:eq(0))/curBatchDim --classify R to R
                            confusion[1][2] = 1-confusion[1][1] --classify R to G
                            confusion[2][2] = torch.sum(dis_result[{{1,curBatchDim}}]:eq(1))/curBatchDim --classify G to G
                            confusion[2][1] = 1-confusion[2][2] --classify G to R

                        end
                        
                        err = GAN_err 
                        tot_error = tot_error + err
                        cnt_error = cnt_error + 1
                        
                        gradParams[{{1,GenParamNum}}]:zero()
                        gradParams[{{GenParamNum+1,-1}}]:div(2*curBatchDim)

                    elseif train_model == "G" then
                        --GenModel training
                        --mse loss
                        MSE_err = mse:forward(output_Gen,targets)/(outputSz*outputSz*curBatchDim)
                        MSE_dfdo = mse:backward(output_Gen,targets)/(outputSz*outputSz*curBatchDim)
                        GenModel:backward(inputs,MSE_dfdo)
                        
                        --classification loss
                        output_Dis = DisModel:forward(output_Gen)
                        Dis_target = torch.CudaTensor(curBatchDim,1):fill(0) --give 0 to G
                        GAN_err = bce:forward(output_Dis,Dis_target)*1e-3
                        GAN_dfdo = bce:backward(output_Dis,Dis_target)*1e-3
                        GAN_dfdi = DisModel:backward(output_Gen,GAN_dfdo)
                        GenModel:backward(inputs,GAN_dfdi)

                        --tv loss
                        --tv_err = tv:forward(output_Gen,nil)/(outputSz*outputSz)*2e-8
                        --tv_dfdo = tv:backward(output_Gen,nil)/(outputSz*outputSz)*2e-8
                        GenModel:backward(inputs,tv_dfdo)
                        
                        err = MSE_err + GAN_err --+ tv_err
                        tot_error = tot_error + err
                        cnt_error = cnt_error + 1
                        
                        gradParams[{{1,GenParamNum}}]:div(2*curBatchDim)
                        gradParams[{{GenParamNum+1,-1}}]:zero()
                    end

                    return err,gradParams
                end
            
            optimMethod(feval, params, optimState)
        end

        tot_iter = tot_iter + 1
                
        if tot_iter == iter_lr_decay then
            optimState.learningRate = optimState.learningRate/10
        end
        
        if tot_iter % 1000 == 0 then
            print("iter: " .. tot_iter .. "/" .. iterLimit .. " batch: " ..  t .. "/" .. trainDataSz .. " loss: " .. tot_error/cnt_error)
        end

    end
   
    if epoch == epochNum then
        local filename = paths.concat(save_dir, GenModelName)
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('==> saving model to '..filename)
        torch.save(filename, GenModel)

        local filename = paths.concat(save_dir, DisModelName)
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('==> saving model to '..filename)
        torch.save(filename, DisModel)

    end
end


