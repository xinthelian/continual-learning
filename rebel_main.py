#!/usr/bin/env python3
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


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Set indicator-dictionary for correctly retrieving / checking input options
    kwargs = {'main': True}
    # Define input options
    parser = options.define_args(filename="main", description='Run an individual continual learning experiment '
                                                              'using the "academic continual learning setting".')
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
    # To see all the parameter settings, use "./main.py -h" in command line.
    # print(args)
    # Namespace(
    #     get_stamp=False, seed=0, cuda=True, save=True, full_stag='none', full_ltag='none', 
    #     train=True, d_dir='./store/datasets', m_dir='./store/models', p_dir='./store/plots', 
    #     r_dir='./store/results', time=False, pdf=False, visdom=False, results_dict=False, 
    #     loss_log=2000, acc_log=2000, acc_n=1024, sample_log=2000, sample_n=64, no_samples=False, 
    #     experiment='splitMNIST', scenario='class', contexts=5, iters=2000, batch=128, normalize=False, 
    #     conv_type='standard', n_blocks=2, depth=0, rl=None, channels=16, conv_bn='yes', conv_nl='relu', 
    #     gp=False, fc_lay=3, fc_units=400, fc_drop=0.0, fc_bn='no', fc_nl='relu', z_dim=100, 
    #     singlehead=False, lr=0.001, optimizer='adam', momentum=0.0, pre_convE=False, convE_ltag='e100', 
    #     seed_to_ltag=False, freeze_convE=False, neg_samples='all', recon_loss='BCE', bce=False, 
    #     bce_distill=False, joint=False, cummulative=False, xdg=False, gating_prop=None, 
    #     separate_networks=False, ewc=False, si=False, ncl=False, ewc_kfac=False, owm=False, 
    #     weight_penalty=False, reg_strength=1.0, precondition=False, alpha=1e-10, 
    #     importance_weighting=None, fisher_n=None, fisher_batch=1, fisher_labels='all', 
    #     fisher_kfac=False, fisher_init=False, data_size=12000, epsilon=0.1, offline=False, gamma=1.0, 
    #     lwf=False, distill=False, temp=2.0, fromp=False, tau=1000.0, budget=100, 
    #     use_full_capacity=False, sample_selection='random', add_buffer=False, replay='none', 
    #     use_replay='normal', agem=False, eps_agem=1e-07, g_z_dim=100, g_fc_lay=3, g_fc_uni=400, 
    #     g_iters=2000, lr_gen=0.001, brain_inspired=False, feedback=False, prior='standard', 
    #     per_class=False, n_modes=1, dg_gates=False, dg_type='class', dg_prop=0.1, hidden=False, 
    #     icarl=False, prototypes=False, gen_classifier=False, eval_s=50, si_c=5000.0, ewc_lambda=1000000000.0)


def run(args, verbose=False):

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

    #-------------------------------------------------------------------------------------------------#

    #----------------#
    #----- DATA -----#
    #----------------#

    # Prepare data for chosen experiment
    if verbose:
        print("\n\n " +' LOAD DATA '.center(70, '*'))
    (train_datasets, test_datasets), config = get_context_set(
        name=args.experiment, scenario=args.scenario, contexts=args.contexts, data_dir=args.d_dir,
        normalize=checkattr(args, "normalize"), verbose=verbose, exception=(args.seed==0),
        singlehead=checkattr(args, 'singlehead'), train_set_per_class=checkattr(args, 'gen_classifier')
    )
    # print("Data:")
    # print(train_datasets, test_datasets, config)
    # Data:
    # [<data.manipulate.TransformedDataset object at 0x13091a6a0>, 
    # <data.manipulate.TransformedDataset object at 0x13091a9a0>, 
    # <data.manipulate.TransformedDataset object at 0x13091ab80>, 
    # <data.manipulate.TransformedDataset object at 0x13091ad60>, 
    # <data.manipulate.TransformedDataset object at 0x13091af40>] 
    # [<data.manipulate.TransformedDataset object at 0x13091a7c0>, 
    # <data.manipulate.TransformedDataset object at 0x13091aa90>, 
    # <data.manipulate.TransformedDataset object at 0x13091ac70>, 
    # <data.manipulate.TransformedDataset object at 0x13091ae50>, 
    # <data.manipulate.TransformedDataset object at 0x130923070>] 
    # {'size': 32, 'channels': 1, 'classes': 10, 'normalize': False, 'classes_per_context': 10, 'output_units': 10}

    print("Train datasets:")
    print(train_datasets)
    print(len(train_datasets))

    print("Example dataset:")
    example_dataset = train_datasets[0]
    print("Number of elements:", len(example_dataset))  # 60000, so there are 60000 datapoints
    print("dataset preview:")
    print(example_dataset.dataset)
    print(type(example_dataset.dataset))  # torchvision.datasets.mnist.MNIST
    print("Single data preview:")
    print(example_dataset[0])  # (torchdata, class)
    print(example_dataset[13])
    print(example_dataset[0][0].shape)  # torch.Size([1, 32, 32])
    example_img = np.array(example_dataset[0][0])
    print(example_img.shape)
    # print(np.array(example_dataset[0][0]))
    # print(example_dataset[0].shape)
    # print(train_datasets[0])
    # print(np.array(example_dataset))


    # The experiments in this script follow the academic continual learning setting,
    # the above lines of code therefore load both the 'context set' and the 'data stream'

    #-------------------------------------------------------------------------------------------------#

    #-----------------------------#
    #----- FEATURE EXTRACTOR -----#
    #-----------------------------#


if __name__ == '__main__':
    # -load input-arguments
    args = handle_inputs()
    # -run experiment
    run(args, verbose=True)