require 'torch'
require 'cunn'
    

torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(os.time())

dofile "etc.lua"
dofile "data.lua"
dofile "model.lua"
dofile "train.lua"
dofile "test.lua"


if mode == "train" then
    
    if continue == true then
        print("model loading...")
        GenModel = torch.load(save_dir .. GenModelName)
        DisModel = torch.load(save_dir .. DisModelName)
    end

    trainData, trainLabel = load_data()
    fp_err = io.open("result/loss_" .. testScale .. ".txt","a")
    fp_PSNR = io.open("result/PSNR_" .. testScale .. ".txt","a")
    while tot_iter <= iterLimit do
        train(trainData, trainLabel)
        err = tot_error/cnt_error
        fp_err:write(err,"\n")
        test()
        fp_PSNR:write(PSNR_sum/testDataSz,"\n")
    end
    fp_err:close()
    fp_PSNR:close()

elseif mode == "test" then
    test()
end
