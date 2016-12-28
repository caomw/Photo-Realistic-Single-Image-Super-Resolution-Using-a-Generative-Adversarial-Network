mode = "test"
GenModelName = "GenModel.net"
DisModelName = "DisModel.net"
continue = false
continue_iter = 0

db_dir = "/media/sda1/Data/ILSVRC/SR_ILSVRC2015_val_4_rgb/"
test_dir = "/media/sda1/Data/SR/Test/BSD100/parsed/"
save_dir = db_dir .. "model_save/"
testDataSz = 100
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
iter_lr_decay = 1e5
iterLimit = 2e5 - continue_iter



function combine_and_flatten_parameters(...)
  local nets = { ... }
  local parameters,gradParameters = {}, {}
  for i=1,#nets do
    local w, g = nets[i]:parameters()
    for i=1,#w do
      table.insert(parameters, w[i])
      table.insert(gradParameters, g[i])
    end
    if i == 1 then
        GenParamNum = nn.Module.flatten(gradParameters):size()[1]
    else
        DisParamNum = nn.Module.flatten(gradParameters):size()[1]
    end

    

  end
  return nn.Module.flatten(parameters), nn.Module.flatten(gradParameters)
end
