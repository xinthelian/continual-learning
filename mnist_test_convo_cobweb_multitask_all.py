import os
import numpy as np
import time
import torch
from torch import optim
# -custom-written libraries
import utils
from utils import checkattr
from data.load import get_context_set
from models import define_models as define
from models.cl.continual_learner import ContinualLearner
from models.cl.memory_buffer import MemoryBuffer
from models.cl import fromp_optimizer
from train.train_task_based import train_cl, train_fromp, train_gen_classifier
from params import options
from params.param_stamp import get_param_stamp, get_param_stamp_from_args, visdom_name
from params.param_values import set_method_options,check_for_errors,set_default_values
from eval import evaluate, callbacks as cb
from visual import visual_plt
from matplotlib import pyplot as plt

from concept_formation.convo_cobweb import ConvoCobwebTree
from concept_formation.visualize import visualize
from tqdm import tqdm


def handle_inputs():
	# Set indicator-dictionary for correctly retrieving / checking input options
	kwargs = {'main': True}
	# Define input options
	parser = options.define_args(
		filename="main",
		description = 'Run an individual continual learning experiment using the "academic continual learning setting".')

    # parser = options.define_args(filename="main",
    # 							 description='Run an individual continual learning experiment using the "academic continual learning setting".')
	parser = options.add_general_options(parser, **kwargs)
	parser = options.add_eval_options(parser, **kwargs)
	parser = options.add_problem_options(parser, **kwargs)
	parser = options.add_model_options(parser, **kwargs)
	parser = options.add_train_options(parser, **kwargs)
	parser = options.add_cl_options(parser, **kwargs)
	# Parse, process and check chosen options
	args = parser.parse_args()
	set_method_options(args)                         # -if a method's "convenience"-option is chosen, select components
	set_default_values(args, also_hyper_params=True) # -set defaults, some are based on chosen scenario / experiment
	check_for_errors(args, **kwargs)                 # -check whether incompatible options are selected
	return args

def datasets(args, verbose=False):

    # Create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if checkattr(args, 'pdf') and not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    # If only want param-stamp, get it printed to screen and exit
    if checkattr(args, 'get_stamp'):
        print(get_param_stamp_from_args(args=args))
        exit()

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")

    # Report whether cuda is used
    if verbose:
        print("CUDA is {}used".format("" if cuda else "NOT(!!) "))

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)


    # --- DATA ---
    # Prepare data for chosen experiment
    if verbose:
    	print("\n\n " +' LOAD DATA '.center(70, '*'))
    (train_datasets, test_datasets), config = get_context_set(
        name=args.experiment, scenario=args.scenario, contexts=args.contexts, data_dir=args.d_dir,
        normalize=checkattr(args, "normalize"), verbose=verbose, exception=(args.seed==0),
        singlehead=checkattr(args, 'singlehead'), train_set_per_class=checkattr(args, 'gen_classifier')
    )
    # train_datasets, test_datasets: list of continual-learning.data.manipulate.TransformedDataset objects
    # The size of these lists is the number of contexts/tasks.
    return train_datasets, test_datasets, config



def data_packaging(dataset):
	# dataset: continual-learning.data.manipulate.TransformedDataset
	# How the TransformedDataset object be like:
	# -----------
	# class TransformedDataset(Dataset):
    # '''To modify an existing dataset with a transform.
    # This is useful for creating different permutations of MNIST without loading the data multiple times.'''

    # def __init__(self, original_dataset, transform=None, target_transform=None):
    #     super().__init__()
    #     self.dataset = original_dataset
    #     self.transform = transform
    #     self.target_transform = target_transform

    # def __len__(self):
    #     return len(self.dataset)

    # def __getitem__(self, index):
    #     (input, target) = self.dataset[index]
    #     if self.transform:
    #         input = self.transform(input)
    #     if self.target_transform:
    #         target = self.target_transform(target)
    #     return (input, target)

    imgs = []
    labels = []
    for i in range(len(dataset)):
    	img = np.array(dataset[i][0])
    	imgs.append(np.reshape(img, (img.shape[1], img.shape[2])))
    	labels.append(dataset[i][1])
    return imgs, labels


def cobweb_training(tree, imgs, labels):
	"""
	Here the test scenario:
	number of contexts: 5
	experiment: permutedMNIST
	scenario: task-IL
	"""

	# tree = ConvoCobwebTree(filter_size=(5,), stride=(1,))
	preds = []

	for i in tqdm(range(len(imgs)), desc="Processing", unit="item"):
		inst = {}
		inst['image_data'] = imgs[i]
		curr = tree.categorize(inst)

		while curr and 'label' not in curr.av_counts:
			curr = curr.parent  
			# shift to the parent unless there is a label for the node. 
			# So the first instance will def be in the root

		if curr:
			pred = curr.predict('label')  # Predict the value for attr 'label'
		# else:
		# 	pred = 0 if isinstance(labels[0], int) else "0"  # set prediction to 0 by default
		# if pred is None:
		# 	pred = 0 if isinstance(labels[0], int) else "0"

		# What if:
		else:
			pred = labels[0] if isinstance(labels[0], int) else "{}".format(labels[0])
		if pred is None:
			pred = labels[0] if isinstance(labels[0], int) else "{}".format(labels[0])
		preds.append(pred)
		inst['label'] = "{}".format(labels[i])
		tree.ifit(inst)

	return preds


def cobweb_testing(tree, imgs, labels):
	preds = []
	for i in tqdm(range(len(imgs)), desc="Processing", unit="item"):
		inst = {}
		inst['image_data'] = imgs[i]
		curr = tree.categorize(inst)
		preds.append(curr.predict('label'))
	return preds


if __name__ == '__main__':
	args = handle_inputs()
	datasets_tr, datasets_te, _ = datasets(args, verbose=True)
	n_contexts = len(datasets_tr)

	tree = ConvoCobwebTree(filter_size=(5,), stride=(1,))

	# Store the data of previous contexts to test the model after every context
	imgs_te_overall = []
	labels_te_overall = []

	for i in range(n_contexts):
		print("\n======== MODEL TRAINING: CONTEXT {} ========".format(i+1))
		dataset_tr = datasets_tr[i]
		dataset_te = datasets_te[i]
		imgs_tr, labels_tr = data_packaging(dataset_tr)
		imgs_te, labels_te = data_packaging(dataset_te)

		# # Daylight version:
		# imgs_tr = imgs_tr[:50]
		# labels_tr = labels_tr[:50]
		# labels_tr = [str(y) for y in labels_tr]
		# imgs_te = imgs_te[:10]
		# labels_te = labels_te[:10]
		# labels_te = [str(y) for y in labels_te]

		# Midnight version:
		# imgs_tr = imgs_tr[:100]
		# labels_tr = labels_tr[:100]
		labels_tr = [str(y) for y in labels_tr]
		# imgs_te = imgs_te[:30]
		# labels_te = labels_te[:30]
		labels_te = [str(y) for y in labels_te]

		imgs_te_overall.append(imgs_te)
		labels_te_overall.append(labels_te)

		preds_tr = cobweb_training(tree, imgs_tr, labels_tr)
		errors_tr = [int(preds_tr[i] != v) for i, v in enumerate(labels_tr)]
		accuracy_tr = 1 - sum(errors_tr) / len(labels_tr)
		print("Training accuracy: {}".format(accuracy_tr))

		# print("\nNow testing the cobweb tree in context {}:".format(i+1))
		# preds_te = cobweb_testing(tree, imgs_te, labels_te)
		# errors_te = [int(preds_te[i] != v) for i, v in enumerate(labels_te)]
		# accuracy_te = 1 - sum(errors_te) / len(labels_te)
		# print("Test accuracy on the current context: {}".format(accuracy_te))
		
		print("\n--------- MODEL TESTING: CONTEXT {} ---------".format(i+1))
		for j in range(i+1):
			print("Now testing the cobweb tree in context {}:".format(j+1))
			imgs_te_prev = imgs_te_overall[j]
			labels_te_prev = labels_te_overall[j]
			preds_te_prev = cobweb_testing(tree, imgs_te_prev, labels_te_prev)
			errors_te_prev = [int(preds_te_prev[i] != v) for i, v in enumerate(labels_te_prev)]
			accuracy_te_prev = 1 - sum(errors_te_prev) / len(labels_te)
			print("Test accuracy on the current context: {}".format(accuracy_te_prev))


	# examples_tr = datasets_tr[0]
	# examples_te = datasets_te[0]
	# # examples_tr = examples_tr[:100]
	# # example_te = datasets_te[0]

	# imgs_tr, labels_tr = data_packaging(examples_tr)
	# imgs_te, labels_te = data_packaging(examples_te)

	# imgs_tr = imgs_tr[:50]
	# labels_tr = labels_tr[:50]
	# labels_tr = [str(y) for y in labels_tr]

	# # imgs_te = imgs_te[:10]
	# # labels_te = labels_te[:10]
	# # labels_te = [str(y) for y in labels_te]

	# tree, preds = cobweb_training(imgs_tr, labels_tr)

	# print("Predicted:")
	# print([label for label in preds])
	# print("Actual:")
	# print([label for label in labels_tr])
	# print("Error:")
	# errors = [int(preds[i] != v) for i, v in enumerate(labels_tr)]
	# print(errors)
	# print("Accuracy:")
	# print(1 - sum(errors) / len(labels_tr))

	# plt.plot(np.mean(errors, 0))
	# plt.show()

	for level in tree.trees:
	    visualize(tree.trees[level])
	    time.sleep(1)

	visualize(tree)


