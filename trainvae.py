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


def train(rxn_trees, model, args):

	random.shuffle(rxn_trees)
	
	model.cuda()

	n = len(rxn_trees)
	ind_list = [i for i in range(n)]

	lr = args['lr']
	batch_size = args['batch_size']
	beta = args['beta']
	val_trees = rxn_trees[:3000]
	train_trees = rxn_trees[3000:-1]
	print("trainng size:", len(train_trees))
	print("valid size:", len(val_trees))
	optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = 0.0001)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.5)
	tr_rec_loss_list = []
	tr_kl_loss_list=[]

	for epoch in range(args['epochs']):
		random.shuffle(train_trees)
		dataloader = DataLoader(train_trees, batch_size = batch_size, shuffle = True, collate_fn = lambda x:x)
		total_loss = 0
		total_kl_loss =0
		total_template_loss = 0
		total_template_acc = 0
		total_molecule_loss = 0
		total_molecule_acc =0

		for it, batch in enumerate(dataloader):
			#print(epoch, it, len(dataloader))
			
			model.zero_grad()
			t_loss, template_loss, molecule_loss, template_acc, molecule_acc, kl_loss = model(batch, beta)
			t_loss.backward()
			optimizer.step()
			total_loss += t_loss
			total_kl_loss += kl_loss

			total_template_loss += template_loss
			total_template_acc += template_acc
			total_molecule_loss += molecule_loss
			total_molecule_acc += molecule_acc

		print("*******************Epoch", epoch, "******************")
		print("training loss")
		print("---> template loss:", total_template_loss.item()/len(dataloader), "tempalte acc:", total_template_acc.item()/len(dataloader))
		print("---> molecule loss:", total_molecule_loss.item()/len(dataloader), "molecule acc:", total_molecule_acc.item()/len(dataloader))
		print("---> kl loss:", total_kl_loss.item()/len(dataloader))
		print("---> total loss:", total_loss.item()/len(dataloader))
		validate(val_trees, model, args)

		if (epoch + 1) % 10==0:
			torch.save(model.state_dict(), args['save_path'] + "/rxnvae_weight.npy")
	#print("---> reconstruction loss:", total_loss.item()/len(dataloader)-beta * total_kl_loss.item()/len(dataloader))

def validate(rxn_trees, model, args):
	beta = args['beta']
	batch_size = args['batch_size']
	dataloader = DataLoader(rxn_trees, batch_size = batch_size, shuffle = True, collate_fn = lambda x:x)
	total_template_loss = 0
	total_template_acc = 0
	total_molecule_distance_loss =0
	#total_molecule_label_loss = 0
	total_molecule_loss = 0
	total_molecule_acc = 0
	total_loss = 0

	with torch.no_grad():
		for it, batch in enumerate(dataloader):
			t_loss, template_loss, molecule_loss, template_acc, molecule_acc, kl_loss = model(batch, beta)
			total_template_acc += template_acc
			total_template_loss += template_loss
			total_molecule_acc += molecule_acc
			total_molecule_loss += molecule_loss
		print(">>> template loss: ",total_template_loss.item()/len(dataloader), "template acc:", total_template_acc/len(dataloader))
		print(">>> molecule loss: ",total_molecule_loss.item()/len(dataloader), "molecule acc:", total_molecule_acc/len(dataloader))
			




parser = OptionParser()
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=50)
parser.add_option("-d", "--depth", dest="depth", default=2)
parser.add_option("-b", "--batch", dest="batch_size", default = 32)
parser.add_option("-s", "--save_dir", dest="save_path", default="weights")
parser.add_option("-t", "--data_path", dest="data_path")
parser.add_option("-q", "--lr", dest="lr", default = 0.001)
parser.add_option("-z", "--beta", dest="beta", default = 1.0)
parser.add_option("-e", "--epochs", dest="epochs", default = 100)

opts, _ = parser.parse_args()

# get parameters
batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
beta = float(opts.beta)
lr = float(opts.lr)
data_filename = opts.data_path
epochs = int(opts.epochs)
save_path = opts.save_path




args={}
args['beta'] = beta
args['lr'] = lr
args['batch_size'] = batch_size
args['datasetname'] = data_filename
args['epochs'] = epochs
args['save_path'] = save_path

print("hidden size:", hidden_size, "latent_size:", latent_size, "batch size:", batch_size, "depth:", depth)
print("beta:", beta, "lr:", lr)
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


if torch.cuda.is_available():
	device = torch.device("cuda")
	torch.cuda.set_device(1)
else:
	device = torch.device("cpu")
print("Running on device:", device)

mpn = MPN(hidden_size, depth)
model = RXNVAE(reactantDic, templateDic, hidden_size, latent_size, depth, reactant_embedding=None, template_embedding=None)
train(rxn_trees, model, args)






























