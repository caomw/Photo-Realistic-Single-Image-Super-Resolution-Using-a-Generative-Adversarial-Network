require 'torch'
require 'nn'
require 'cudnn'
require 'module/normalConv'
require 'module/normalLinear'
require 'module/normalDeconv'
dofile 'etc.lua'


GenModel = nn.Sequential()
kernelSz = 3
prev_fDim = inputDim
next_fDim = 64
GenModel:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
GenModel:add(nn.ReLU(true))

for lid = 1,n do
    
    concat = nn.ConcatTable()
    concat:add(nn.Identity())
    subGenModel = nn.Sequential()

    kernelSz = 3
    prev_fDim = next_fDim
    next_fDim = 64
    subGenModel:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
    subGenModel:add(nn.SpatialBatchNormalization(next_fDim))
    subGenModel:add(nn.ReLU(true))

    subGenModel:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
    subGenModel:add(nn.SpatialBatchNormalization(next_fDim))

    concat:add(subGenModel)
    GenModel:add(concat)
    GenModel:add(nn.CAddTable(false))

end

kernelSz = 4
prev_fDim = next_fDim
next_fDim = 64
GenModel:add(cudnn.normalDeconv(prev_fDim,next_fDim,kernelSz,kernelSz,2,2,(kernelSz-1)/2,(kernelSz-1)/2,1,1,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
GenModel:add(nn.ReLU(true))

GenModel:add(cudnn.normalDeconv(prev_fDim,next_fDim,kernelSz,kernelSz,2,2,(kernelSz-1)/2,(kernelSz-1)/2,1,1,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
GenModel:add(nn.ReLU(true))

kernelSz = 17
prev_fDim = next_fDim
next_fDim = outputDim
GenModel:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))


filename = paths.concat(save_dir, "SRResNet.net")
GenModel = torch.load(filename)
-------------------------------

DisModel = nn.Sequential()

kernelSz = 3
prev_fDim = inputDim
next_fDim = 64
DisModel:add(cudnn.normalDeconv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,1,1,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
DisModel:add(nn.RReLU())

    
kernelSz = 3
prev_fDim = next_fDim
next_fDim = 64
DisModel:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,2,2,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
DisModel:add(nn.RReLU())
DisModel:add(nn.SpatialBatchNormalization(next_fDim))

kernelSz = 3
prev_fDim = next_fDim
next_fDim = 128
DisModel:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
DisModel:add(nn.RReLU())
DisModel:add(nn.SpatialBatchNormalization(next_fDim))

kernelSz = 3
prev_fDim = next_fDim
next_fDim = 128
DisModel:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,2,2,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
DisModel:add(nn.RReLU())
DisModel:add(nn.SpatialBatchNormalization(next_fDim))

kernelSz = 3
prev_fDim = next_fDim
next_fDim = 256
DisModel:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
DisModel:add(nn.RReLU())
DisModel:add(nn.SpatialBatchNormalization(next_fDim))

kernelSz = 3
prev_fDim = next_fDim
next_fDim = 256
DisModel:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,2,2,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
DisModel:add(nn.RReLU())
DisModel:add(nn.SpatialBatchNormalization(next_fDim))

kernelSz = 3
prev_fDim = next_fDim
next_fDim = 512
DisModel:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
DisModel:add(nn.RReLU())
DisModel:add(nn.SpatialBatchNormalization(next_fDim))


kernelSz = 3
prev_fDim = next_fDim
next_fDim = 512
DisModel:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,2,2,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
DisModel:add(nn.RReLU())
DisModel:add(nn.SpatialBatchNormalization(next_fDim))

prev_fDim = 512
next_fDim = next_fDim*outputSz*outputSz
DisModel:add(nn.View(next_fDim):setNumInputDims(3))
DisModel:add(nn.normalLinear(next_fDim,1024,0,math.sqrt(2/next_fDim)))
DisModel:add(nn.RReLu())
DisModel:add(nn.normalLinear(1024,1,0,math.sqrt(2/1024)))
DisModel:add(nn.Sigmoid())

mse = nn.MSECriterion()
bce = nn.BCEVriterion()

cudnn.convert(GenModel, cudnn)
cudnn.convert(DisModel, cudnn)

GenModel:cuda()
DisModel:cuda()
mse:cuda()
bce:cuda()
cudnn.fastest = true
cudnn.benchmark = true
