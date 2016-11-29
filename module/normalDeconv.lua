require 'nn'
require 'cudnn'

do

    local SpatialFullConvolution, parent = torch.class('cudnn.normalDeconv', 'cudnn.SpatialFullConvolution')
    
    -- override the constructor to have the additional range of initialization
    function SpatialFullConvolution:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, adjW, adjH, mean, std)
        parent.__init(self,nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH, adjW, adjH)
                
        self:reset(mean,std)
    end
    
    -- override the :reset method to use custom weight initialization.        
    function SpatialFullConvolution:reset(mean,stdv)
        
        if mean and stdv then
            self.weight:normal(mean,stdv)
            self.bias:zero()
        else
            self.weight:normal(0,1)
            self.bias:zero()
        end
    end

end
