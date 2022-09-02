import sys
sys.path.append('./rxnvae')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser
from collections import deque

from reaction_utils import read_multistep_rxns
from reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates,stats
from vae import RXNVAE, set_batch_nodeID
from mpn import MPN,PP,Discriminator
import random
from evaluate import Evaluator

parser = OptionParser()
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=50)
parser.add_option("-d", "--depth", dest="depth", default=2)
parser.add_option("-b", "--batch", dest="batch_size", default = 32)
parser.add_option("-s", "--save_dir", dest="save_path")
parser.add_option("-t", "--data_path", dest="data_path")
parser.add_option("-o", "--output_file", dest="output_file", default = "Results/sampled_rxns.txt")


opts, _ = parser.parse_args()

# get parameters
batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
data_filename = opts.data_path
w_save_path = opts.save_path
output_file = opts.output_file

if torch.cuda.is_available():
	#device = torch.device("cuda:1")
	device = torch.device("cuda")
	torch.cuda.set_device(1)
else:
	device = torch.device("cpu")

print("hidden size:", hidden_size, "latent_size:", latent_size, "depth:", depth)
print("loading data.....")
data_filename = opts.data_path
routes, scores = read_multistep_rxns(data_filename)
rxn_trees = [ReactionTree(route) for route in routes]
molecules = [rxn_tree.molecule_nodes[0].smiles for rxn_tree in rxn_trees]
reactants = extract_starting_reactants(rxn_trees)
templates, n_reacts = extract_templates(rxn_trees)
reactantDic = StartingReactants(reactants)
templateDic = Templates(templates, n_reacts)

print("size of reactant dic:", reactantDic.size())
print("size of template dic:", templateDic.size())
#print(templateDic.template_list)


# loading model

mpn = MPN(hidden_size, depth)
model = RXNVAE(reactantDic, templateDic, hidden_size, latent_size, depth, reactant_embedding=None, template_embedding=None)
checkpoint = torch.load(w_save_path, map_location=device)
model.load_state_dict(checkpoint)
print("loaded model....")
evaluator = Evaluator(latent_size, model)
#evaluator.validate_and_save(rxn_trees, output_file=output_file)
#evaluator.novelty_and_uniqueness(["./Results/qed1.txt"], rxn_trees)
evaluator.novelty_and_uniqueness(["./Results/qed1.txt"], rxn_trees)
#evaluator.kde_plot(["./Results/generated_rxns.txt"],["./Results/qed1.txt"], metric="qed")


