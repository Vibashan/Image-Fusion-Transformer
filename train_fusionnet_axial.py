# Training a IFT network

import os

import sys
import time
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
import pdb
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import utils
from kornia.filters.sobel import sobel
from net import NestFuse_light2_nodense, Fusion_network, RFN_decoder
from checkpoint import load_checkpoint
from args_fusion import args
import pytorch_msssim
#from vit_model import VisionTransformer

import warnings
warnings.filterwarnings("ignore")

EPSILON = 1e-5

def main():
	original_imgs_path, _ = utils.list_images(args.dataset_ir)
	train_num = 80000
	original_imgs_path = original_imgs_path[:train_num]
	random.shuffle(original_imgs_path)
	# True - RGB , False - gray
	img_flag = False
	alpha_list = [700]
	w_all_list = [[6.0, 3.0]]

	for w_w in w_all_list:
		w1, w2 = w_w
		for alpha in alpha_list:
			train(original_imgs_path, img_flag, alpha, w1, w2)


def train(original_imgs_path, img_flag, alpha, w1, w2):

	batch_size = args.batch_size
	# load network model
	nc = 1
	input_nc = nc
	output_nc = nc
	#nb_filter = [64, 128, 256, 512]
	nb_filter = [64, 112, 160, 208]
	f_type = 'res'

	
	with torch.no_grad():
		deepsupervision = False
		nest_model = NestFuse_light2_nodense(nb_filter, input_nc, output_nc, deepsupervision)
		model_path = args.resume_nestfuse
		# load auto-encoder network
		print('Resuming, initializing auto-encoder using weight from {}.'.format(model_path))
		nest_model.load_state_dict(torch.load(model_path))
		nest_model.cuda()
		nest_model.eval()

	# fusion network
	fusion_model = Fusion_network(nb_filter, f_type)
	fusion_model.cuda()
	fusion_model.train()

	if args.resume_fusion_model is not None:
		print('Resuming, initializing fusion net using weight from {}.'.format(args.resume_fusion_model))
		fusion_model.load_state_dict(torch.load(args.resume_fusion_model))
	optimizer = Adam(fusion_model.parameters(), args.lr)
	mse_loss = torch.nn.MSELoss()
	ssim_loss = pytorch_msssim.msssim


	tbar = trange(args.epochs)
	print('Start training.....')
	mode = args.mode
	print(mode)
	# creating save path
	temp_path_model = os.path.join(args.save_fusion_model)
	temp_path_loss  = os.path.join(args.save_loss_dir)
	if os.path.exists(temp_path_model) is False:
		os.mkdir(temp_path_model)

	if os.path.exists(temp_path_loss) is False:
		os.mkdir(temp_path_loss)

	temp_path_model_w = os.path.join(args.save_fusion_model, str(w1), mode)
	temp_path_loss_w  = os.path.join(args.save_loss_dir, str(w1))
	if os.path.exists(temp_path_model_w) is False:
		os.mkdir(temp_path_model_w)

	if os.path.exists(temp_path_loss_w) is False:
		os.mkdir(temp_path_loss_w)

	Loss_feature = []
	Loss_ssim = []
	Loss_all = []
	count_loss = 0
	all_ssim_loss = 0.
	all_fea_loss = 0.
	sobel_loss = nn.L1Loss()
	for e in tbar:
		print('Epoch %d.....' % e)
		# load training database
		image_set_ir, batches = utils.load_dataset(original_imgs_path, batch_size)
		count = 0
		nest_model.cuda()
		#trans_model.cuda()
		fusion_model.cuda()
		for batch in range(batches):
			image_paths_ir = image_set_ir[batch * batch_size:(batch * batch_size + batch_size)]
			img_ir = utils.get_train_images(image_paths_ir, height=args.HEIGHT, width=args.WIDTH, flag=img_flag)

			image_paths_vi = [x.replace('lwir', 'visible') for x in image_paths_ir]
			img_vi = utils.get_train_images(image_paths_vi, height=args.HEIGHT, width=args.WIDTH, flag=img_flag)

			count += 1
			optimizer.zero_grad()

			img_ir = Variable(img_ir, requires_grad=False)
			img_vi = Variable(img_vi, requires_grad=False)

			img_ir = img_ir.cuda()
			img_vi = img_vi.cuda()

			# encoder
			en_ir = nest_model.encoder(img_ir)
			en_vi = nest_model.encoder(img_vi)
			# fusion
			f = fusion_model(en_ir, en_vi)
			# decoder
			outputs = nest_model.decoder_eval(f)

			# resolution loss: between fusion image and visible image
			x_ir = Variable(img_ir.data.clone(), requires_grad=False)
			x_vi = Variable(img_vi.data.clone(), requires_grad=False)

			######################### LOSS FUNCTION #########################
			loss1_value = 0.
			loss2_value = 0.
			for output in outputs:
				output = (output - torch.min(output)) / (torch.max(output) - torch.min(output) + EPSILON)
				output = output * 255
				# ---------------------- LOSS IMAGES ------------------------------------
				# detail loss
				ssim_loss_temp2 = ssim_loss(output, x_vi, normalize=True)
				loss1_value += alpha * (1 - ssim_loss_temp2)

				# feature loss
				g2_ir_fea = en_ir
				g2_vi_fea = en_vi
				g2_fuse_fea = f

				w_ir = [w1, w1, w1, w1]
				w_vi = [w2, w2, w2, w2]
				w_fea = [1, 10, 100, 1000]
				for ii in range(4):
					g2_ir_temp = g2_ir_fea[ii]
					g2_vi_temp = g2_vi_fea[ii]
					g2_fuse_temp = g2_fuse_fea[ii]
					(bt, cht, ht, wt) = g2_ir_temp.size()
					loss2_value += w_fea[ii]*mse_loss(g2_fuse_temp, w_ir[ii]*g2_ir_temp + w_vi[ii]*g2_vi_temp)

			loss1_value /= len(outputs)
			loss2_value /= len(outputs)

			total_loss = loss1_value + loss2_value 
			total_loss.backward()
			optimizer.step()

			all_fea_loss += loss2_value.item() # 
			all_ssim_loss += loss1_value.item() # 
			if (batch + 1) % args.log_interval == 0:
				mesg = "{}\t Alpha: {} \tW-IR: {}\tEpoch {}:\t[{}/{}]\t ssim loss: {:.6f}\t fea loss: {:.6f}\t total: {:.6f}".format(
					time.ctime(), alpha, w1, e + 1, count, batches,
								  all_ssim_loss / args.log_interval,
								  all_fea_loss / args.log_interval,
								  (all_fea_loss + all_ssim_loss) / args.log_interval
				)
				tbar.set_description(mesg)
				Loss_ssim.append( all_ssim_loss / args.log_interval)
				Loss_feature.append(all_fea_loss / args.log_interval)
				Loss_all.append((all_fea_loss + all_ssim_loss) / args.log_interval)
				count_loss = count_loss + 1
				all_ssim_loss = 0.
				all_fea_loss = 0.

		# save model
		save_model_filename = mode + ".model"
		save_model_path = os.path.join(temp_path_model_w, save_model_filename)
		torch.save(fusion_model.state_dict(), save_model_path)

		print("\nDone, trained model saved at", save_model_path)


def check_paths(args):
	try:
		if not os.path.exists(args.vgg_model_dir):
			os.makedirs(args.vgg_model_dir)
		if not os.path.exists(args.save_model_dir):
			os.makedirs(args.save_model_dir)
	except OSError as e:
		print(e)
		sys.exit(1)


if __name__ == "__main__":
	main()
