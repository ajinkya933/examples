from __future__ import print_function
import argparse
import torch
from PIL import Image
from torchvision.transforms import ToTensor
import os

import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, required=True, help='input image to use')
parser.add_argument('--model', type=str, required=True, help='model file to use')
parser.add_argument('--output_filename', type=str, help='where to save the output image')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

#print(opt)
print('input_path', opt.input_image)
print('output_path',opt.output_filename)


for file in os.listdir(opt.input_image):
	if file.endswith('.jpg'or'.png'):
		i = Image.open(os.path.join(opt.input_image, file))
		img = i.convert('YCbCr')
		y, cb, cr = img.split()
		model = torch.load(opt.model)
		img_to_tensor = ToTensor()
		input = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])
		out = model(input)
		out = out.cpu()
		out_img_y = out[0].detach().numpy()
		out_img_y *= 255.0
		out_img_y = out_img_y.clip(0, 255)
		out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
		out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
		out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
		out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
		out_img.save(os.path.join(opt.output_filename, file))
