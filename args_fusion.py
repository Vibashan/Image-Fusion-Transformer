
class args():

	# training args
	epochs = 2 #"number of training epochs, default is 2"
	batch_size = 4 #"batch size for training, default is 4"
	dataset_ir = "./data/kaist-rgbt/images"
	dataset_vi = "./data/kaist-rgbt/images"

	dataset = 'medical'
	#dataset = 'focus'

	HEIGHT = 256
	WIDTH = 256

	save_fusion_model = "models/train/fusionnet/"
	save_loss_dir = './models/train/loss_fusionnet/'

	image_size = 256 #"size of training images, default is 256 X 256"
	cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"
	seed = 42 #"random seed for training"

	lr = 1e-4 #"learning rate, default is 0.001"
	log_interval = 10 #"number of images after which the training loss is logged, default is 500"
	resume_fusion_model = None
	# nest net model
	resume_nestfuse = './models/nestfuse/nestfuse_gray_1e2.model'
	resume_vit = './imagenet21k+imagenet2012_ViT-L_16.pth'
	fusion_model = './models/rfn_twostage/'

	mode = "fusion_axial"



