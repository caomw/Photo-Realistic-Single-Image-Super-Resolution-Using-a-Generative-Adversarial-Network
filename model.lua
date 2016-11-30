require 'torch'
require 'nn'
require 'cudnn'
require 'module/normalConv'
require 'module/normalLinear'
require 'module/normalDeconv'
dofile 'etc.lua'


model = nn.Sequential()
kernelSz = 3
prev_fDim = inputDim
next_fDim = 64
model:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
model:add(nn.ReLU(true))

for lid = 1,n do
    
    concat = nn.ConcatTable()
    concat:add(nn.Identity())
    subModel = nn.Sequential()

    kernelSz = 3
    prev_fDim = next_fDim
    next_fDim = 64
    subModel:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
    subModel:add(nn.SpatialBatchNormalization(next_fDim))
    subModel:add(nn.ReLU(true))

    subModel:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
    subModel:add(nn.SpatialBatchNormalization(next_fDim))

    concat:add(subModel)
    model:add(concat)
    model:add(nn.CAddTable(false))

end

kernelSz = 4
prev_fDim = next_fDim
next_fDim = 64
model:add(cudnn.normalDeconv(prev_fDim,next_fDim,kernelSz,kernelSz,2,2,(kernelSz-1)/2,(kernelSz-1)/2,1,1,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
model:add(nn.ReLU(true))

model:add(cudnn.normalDeconv(prev_fDim,next_fDim,kernelSz,kernelSz,2,2,(kernelSz-1)/2,(kernelSz-1)/2,1,1,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))
model:add(nn.ReLU(true))

kernelSz = 17
prev_fDim = next_fDim
next_fDim = outputDim
model:add(cudnn.normalConv(prev_fDim,next_fDim,kernelSz,kernelSz,1,1,(kernelSz-1)/2,(kernelSz-1)/2,0,math.sqrt(2/(kernelSz*kernelSz*prev_fDim))))

criterion = nn.MSECriterion()

cudnn.convert(model, cudnn)
cudnn.convert(VGGNet, cudnn)

model:cuda()
VGGNet:cuda()
criterion:cuda()
cudnn.fastest = true
cudnn.benchmark = true

