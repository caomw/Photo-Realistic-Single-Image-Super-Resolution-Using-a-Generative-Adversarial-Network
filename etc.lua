mode = "train"
modelName = "model.net"
loss = "mse" --"mse" or "vgg"
continue = false
continue_iter = 1e5

db_dir = "/home/mks0601/workspace/Data/SR_ILSVRC2015_val_4_rgb/"
test_dir = "/home/mks0601/workspace/Data/SR/"
save_dir = db_dir .. "model_save/"
testDataSz = 5
trainScale = {4}
testScale = 4

poolFactor = 4
outputSz = 96
inputSz = outputSz/poolFactor
inputDim = 3
outputDim = 3
n = 15

lr = 1e-4
b1 = 9e-1
batchSz = 16
iterLimit = 1e6 - continue_iter

