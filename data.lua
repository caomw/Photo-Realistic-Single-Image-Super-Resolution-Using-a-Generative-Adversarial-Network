require "torch"
require "image"
dofile "etc.lua"


function load_data()
    
    local input = {}
    local target = {} 
    
    --[===[
    for fid = 1,trainDataSz do    
        
        LRfileName = db_dir .. "SR_ILSVRC2015_val_4_LR_rgb/" .. "img" .. string.format('%06d',fid) .. "_LR.png"
        GTfileName = db_dir .. "SR_ILSVRC2015_val_4_GT_rgb/" .. "img" .. string.format('%06d',fid) .. "_GT.png"

        table.insert(input,LRfileName)
        table.insert(target,GTfileName)
    end
    --]===]
    
    f = io.popen('ls ' .. db_dir .. "SR_ILSVRC2015_val_4_LR/")
    for name in f:lines() do table.insert(input,db_dir .. "SR_ILSVRC2015_val_4_LR/" .. name) end

    f = io.popen('ls ' .. db_dir .. "SR_ILSVRC2015_val_4_GT/")
    for name in f:lines() do table.insert(target,db_dir .. "SR_ILSVRC2015_val_4_GT/" .. name) end

    trainDataSz = table.getn(input)

    return input, target
end

