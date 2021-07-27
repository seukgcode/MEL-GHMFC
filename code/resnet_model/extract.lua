require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'
require 'hdf5'
require 'xlua'
local cjson = require 'cjson'
local t = require 'resnet_model/transforms'


if #arg < 2 then
   io.stderr:write('Usage (Single file mode): th extract-features.lua [MODEL] [FILE] ... \n')
   io.stderr:write('Usage (Batch mode)      : th extract-features.lua [MODEL] [BATCH_SIZE] [DIRECTORY_CONTAINING_IMAGES]  \n')
   os.exit(1)
end

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('--path_resnet', '', 'Path to the resnet model.')
cmd:option('--path_img_file_or_dir', '', 'Single mode: path to img file. Batch mode: dir')
cmd:option('--path_prepro', '', 'Single mode: path to img file. Batch mode: dir')

cmd:option('--out_name', 'img_features', 'output name')


cmd:option('--batch_mode', 0, "Is batch mode? 0: single (default); 1: batch")
cmd:option('--batch_size', 32, 'batch_size')

cmd:option('--gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('--backend', 'cudnn', 'nn|cudnn')

opt = cmd:parse(arg)
print(opt)

--cutorch.setDevice(opt.gpuid)

-- get the list of files
local list_of_filenames = {}
local batch_size = 1

if not paths.filep(opt.path_resnet) then
    io.stderr:write('Model file not found at ' .. arg[1] .. '\n')
    os.exit(1)
end


if opt.batch_mode == 1 then -- batch mode ; collect file from directory

    local lfs  = require 'lfs'
    batch_size = tonumber(opt.batch_size)
    dir_path   = opt.path_img_file_or_dir

    for file in lfs.dir(dir_path) do -- get the list of the files
        if file~="." and file~=".." then
            table.insert(list_of_filenames, dir_path..'/'..file)
        end
    end

else -- single file mode ; collect file from command line
    f = opt.path_img_file_or_dir
    if not paths.filep(f) then
      io.stderr:write('Single mode: file not found: ' .. f .. '\n')
      os.exit(1)
    else
       table.insert(list_of_filenames, f)
    end
end

--cutorch.setDevice(opt.gpuid)
local model = torch.load(opt.path_resnet)

for i = 14,12,-1 do
    model:remove(i)
end
print(model)
model:evaluate()
model=model:cuda()

-- The model was trained with this input normalization
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}

local transform = t.Compose{
   t.ColorNormalize(meanstd),
}

imloader={}
function imloader:load(fname)
    self.im=image.load(fname)
end
function loadim(imname)

    imloader:load(imname,  3, 'float')
    im=imloader.im
    im = image.scale(im, 448, 448)

    if im:size(1)==1 then
        im2=torch.cat(im,im,1)
        im2=torch.cat(im2,im,1)
        im=im2
    elseif im:size(1)==4 then
        im=im[{{1,3},{},{}}]
    end

    im = transform(im)
    return im
end

local sz = #list_of_filenames
if batch_size > sz then batch_size = sz end
local features = torch.FloatTensor(sz, 14, 14, 2048)
print(string.format('processing %d images...',sz))

for i=1,sz,batch_size do
    xlua.progress(i, sz)
    r=math.min(sz,i+batch_size-1)
    local ims=torch.CudaTensor(r-i+1,3,448,448)
    for j=1,r-i+1 do
        ims[j]=loadim(list_of_filenames[i+j-1]):cuda()
    end
    local output = model:forward(ims)
    features[{{i,r},{}}]=output:permute(1,3,4,2):contiguous():float()
    collectgarbage()
end

local h5_file = hdf5.open(opt.path_prepro..'/'..opt.out_name..'.h5', 'w')
h5_file:write("features", features)
h5_file:close()
--h5_file:write("image_list", list_of_filenames)

local text = cjson.encode(list_of_filenames)
local file = io.open(opt.path_prepro..'/img_list.json', 'w')
file:write(text)
file:close()