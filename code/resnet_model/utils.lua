local cjson = require 'cjson'
local utils = {}
require 'nn'
local t = require 'resnet/transforms'

function utils.read_json(path)
    local file = io.open(path, 'r')
    local text = file:read("*a")
    file:close()
    local info = cjson.decode(text)
    return info
end

function utils.write_json(path, j)

    local text = cjson.encode(j)
    local file = io.open(path, 'w')
    file:write(text)
    file:close()
end

-- The model was trained with this input normalization
local meanstd = {
    mean = { 0.485, 0.456, 0.406 },
    std = { 0.229, 0.224, 0.225 },
}

utils.transform = t.Compose{
    t.Scale(256),
    t.ColorNormalize(meanstd),
    t.CenterCrop(224),
}

return utils