
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Image transforms for data augmentation and input normalization
--
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
require 'torch'
require 'paths'
require 'cudnn'
require 'cunn'
require 'image'
require 'nn'

local utils = require 'resnet/utils'
local imagenetLabel = require 'resnet/imagenet'
--
--if #arg < 2 then
--   io.stderr:write('Usage: th object_classify.lua [MODEL] [FILE]...\n')
--   os.exit(1)
--end
--for _, f in ipairs(arg) do
--   if not paths.filep(f) then
--      io.stderr:write('file not found: ' .. f .. '\n')
--      os.exit(1)
--   end
--end


function parse_args()
   cmd = torch.CmdLine()
   cmd:option('--path_qs', 'data/FVQA/all_qs_dict_release.json', 'Path to FVQA qs dict.')
   cmd:option('--path_model', '', 'PATH to the CV model')
   cmd:option('--dir_image', '', 'Dir of the images')
   cmd:option('--dir_output', 'data/vis_FVQA', "Dir of the outputs")
   cmd:option('--threshold', 0.55, 'Threshold of the output')
   return cmd
end

cmd = parse_args()
local opts = cmd:parse(arg)
print(opts)

-- Load the model
local model = torch.load(opts.path_model):cuda()
local softMaxLayer = cudnn.SoftMax():cuda()

-- add Softmax layer
model:add(softMaxLayer)

-- Evaluate mode
model:evaluate()

local N = 50

-- load FVQA qs dict
local qs = utils.read_json(opts.path_qs)
local ans = {}
for k, v in pairs(qs) do
   local img = image.load(opts.dir_image..'/'..v.img_file, 3, 'float')
   img = utils.transform(img)
   -- View as mini-batch of size 1
   local batch = img:view(1, table.unpack(img:size():totable()))
   -- Get the output of the softmax
   local output = model:forward(batch:cuda()):squeeze()

   -- Get the top 5 class indexes and probabilities
   local probs, indexes = output:topk(N, true, true)
   local classes = {}
   for n=1,N do
      if probs[n] >= opts.threshold then
         table.insert(classes, imagenetLabel[indexes[n]])
      else
         break
      end
         --print(probs[n], imagenetLabel[indexes[n]])
      end
   ans[k] = classes
end

utils.write_json(opts.dir_output.."/img1k.json", ans)