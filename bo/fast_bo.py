import sys
sys.path.append('../rxnvae')
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import QED, Descriptors, rdmolops

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

import math, random, sys
from optparse import OptionParser
from collections import deque

from reaction_utils import get_mol_from_smiles, get_smiles_from_mol,read_multistep_rxns, get_template_order, get_qed_score,get_clogp_score
from reaction import ReactionTree, extract_starting_reactants, StartingReactants, Templates, extract_templates,stats
from vae import RXNVAE, set_batch_nodeID
from mpn import MPN,PP,Discriminator
from evaluate import Evaluator
import random
import numpy as np
import networkx as nx

from sparse_gp import SparseGP
import scipy.stats as sps
import sascorer


def decode_many_times(model, latent):
	prob_decode = True
	product_list=[]
	for i in range(50):
		product, reactions = model.rxn_decoder.decode(latent, prob_decode)
		if product != None:
			product_list.append([product, reactions])
	if len(product_list) == 0:
		return None
	else:
		return product_list

def run_bo(X_train, y_train, X_test, y_test, model, parameters, metric, randseed):
	random_seed = int(randseed)
	np.random.seed(random_seed)
	if metric =="logp":
		logp_m = parameters[0]
		logp_s = parameters[1]
		sascore_m = parameters[2]
		sascore_s = parameters[3]
		cycle_m = parameters[4]
		cycle_s = parameters[5]

	filename = "../Results/" + metric + str(random_seed) + ".txt"

	#print("maxmimum score :", np.min(y_train), X_train.shape)
	#print(y_train)
	with open(filename, "w") as writer:
		iteration = 0
		latents = []
		min_scores = []
		while iteration < 5:
			# fit the GP
			#print("maxmimum score :", np.min(y_train), X_train.shape)
			print(iteration)
			np.random.seed(iteration * random_seed)
			M = 500
			sgp = SparseGP(X_train, 0 * X_train, y_train, M)
			sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0, y_test, minibatch_size = 10 * M, max_iterations = 100, learning_rate = 0.001)

			pred, uncert = sgp.predict(X_test, 0 * X_test)
			error = np.sqrt(np.mean((pred - y_test)**2))
			testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
			print('Test RMSE: ', error, ' Test ll: ', testll)

			pred, uncert = sgp.predict(X_train, 0 * X_train)
			error = np.sqrt(np.mean((pred - y_train)**2))
			trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
			print( 'Train RMSE: ', error, 'Train ll: ', trainll)
			#print( 'Train ll: ', trainll)

			next_inputs, values = sgp.batched_greedy_ei(60, np.min(X_train, 0), np.max(X_train, 0))
			valid_smiles =[]
			new_features =[]
			full_rxn_strs=[]
			values = values.flatten()
			count =0
			for i in range(60):
				#print(i)
				latent = next_inputs[i].reshape((1,-1))
				#res = model.decode_many_times(torch.from_numpy(latent).float(), 50)
				res= decode_many_times(model, torch.from_numpy(latent).float())
				if res is not None:
					smiles_list = [re[0] for re in res]
					n_reactions = [len(re[1].split(" ")) for re in res]
					#print(n_reactions)
					for re in res:
						smiles = re[0]
						if len(re[1].split(" ")) > 0 and smiles not in valid_smiles:
							count +=1
							print(smiles, re[1].split(" "), count, i)
							valid_smiles.append(smiles)
							new_features.append(latent)
							full_rxn_strs.append(re[1])
				#print(i, res)
					
			#new_features = np.vstack(new_features)
			scores =[]
			b_valid_smiles=[]
			b_full_rxn_strs=[]
			b_scores=[]
			b_new_features=[]
			for i in range(len(valid_smiles)):
				if metric =="logp":
					mol = rdkit.Chem.MolFromSmiles(valid_smiles[i])
					if mol is None:
						continue
					current_log_P_value = Descriptors.MolLogP(mol)
					current_SA_score = -sascorer.calculateScore(mol)
					cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(mol)))
					if len(cycle_list) == 0:
						cycle_length = 0
					else:
						cycle_length = max([ len(j) for j in cycle_list ])
					if cycle_length <= 6:
						cycle_length = 0
					else:
						cycle_length = cycle_length - 6
					current_cycle_score = -cycle_length
					current_SA_score_normalized = (current_SA_score - sascore_m) / sascore_s
					current_log_P_value_normalized = (current_log_P_value - logp_m) / logp_s
					current_cycle_score_normalized = (current_cycle_score - cycle_m) / cycle_s
					score = current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized
					scores.append(-score)
					b_valid_smiles.append(valid_smiles[i])
					b_full_rxn_strs.append(full_rxn_strs[i])
					b_new_features.append(new_features[i])
				if metric=="qed":
					mol = rdkit.Chem.MolFromSmiles(valid_smiles[i])
					if mol!=None:
						score = QED.qed(mol)
						scores.append(-score)
						b_valid_smiles.append(valid_smiles[i])
						b_full_rxn_strs.append(full_rxn_strs[i])
						b_new_features.append(new_features[i])
			new_features = np.vstack(b_new_features)
			if len(new_features) > 0:
				X_train = np.concatenate([ X_train, new_features ], 0)
				y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)
			iteration+=1

			for i in range(len(b_valid_smiles)):
				line = " ".join([b_valid_smiles[i], b_full_rxn_strs[i], str(scores[i])])
				writer.write(line + "\n")
			#print(iteration, min(scores))

def save(elements, filename):
	with open(filename, 'w') as fp:
		for e in elements:
			fp.write("%s\n" % e)
		print('done')
def read(filename, number=False):
	elements=[]
	with open(filename, 'r') as fp:
		for line in fp:
			e = line[:-1]
			if number:
				e = int(e)
			elements.append(e)
	return elements

parser = OptionParser()
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=50)
parser.add_option("-d", "--depth", dest="depth", default=2)
parser.add_option("-s", "--save_dir", dest="save_path")
parser.add_option("-t", "--data_path", dest="data_path")
parser.add_option("-m", "--metric", dest="metric")
parser.add_option("-r", "--seed", dest="seed", default=1)
opts, _ = parser.parse_args()

# get parameters
hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
data_filename = opts.data_path
w_save_path = opts.save_path
metric = opts.metric
seed = int(opts.seed)


# load model
if torch.cuda.is_available():
	#device = torch.device("cuda:1")
	device = torch.device("cuda")
	torch.cuda.set_device(1)
else:
	device = torch.device("cpu")


print("hidden size:", hidden_size, "latent_size:", latent_size, "depth:", depth)
print("loading data.....")
data_filename = opts.data_path

#save(reactants, "reactants.txt")
#save(templates, "templates.txt")
#save(n_reacts, "n_reacts.txt")

reactants = read("reactants.txt")
templates = read("templates.txt")
n_reacts = read("n_reacts.txt", number=True)

reactantDic = StartingReactants(reactants)
templateDic = Templates(templates, n_reacts)
print("size of reactant dic:", reactantDic.size())
print("size of template dic:", templateDic.size())

routes, scores = read_multistep_rxns(data_filename)
n = len(routes)
ind_list = [i for i in range(n)]
random.shuffle(ind_list)


# shuffle data and use a subset to perform BO (reduce the computational complexity)
routes = [routes[i] for i in range(n)]
scores = [scores[i] for i in range(n)]
routes = routes[:10000]
scores = scores[:10000]





rxn_trees = [ReactionTree(route) for route in routes]
molecules = [rxn_tree.molecule_nodes[0].smiles for rxn_tree in rxn_trees]


# loading model

mpn = MPN(hidden_size, depth)
model = RXNVAE(reactantDic, templateDic, hidden_size, latent_size, depth, reactant_embedding=None, template_embedding=None)
checkpoint = torch.load(w_save_path, map_location=device)
model.load_state_dict(checkpoint)
print("finished loading model...")




latent_list=[]
score_list=[]
print("num of samples:", len(rxn_trees))
latent_list =[]
score_list=[]
if metric =="qed":
	for i, rxn_tree in enumerate(rxn_trees):
		latent = model.encode([rxn_tree])
		#print(i, latent.size(), latent)
		latent_list.append(latent[0])
		smiles = rxn_tree.molecule_nodes[0].smiles
		score_list.append(get_qed_score(smiles))
if metric =="logp":
	logP_values = np.loadtxt('logP_values.txt')
	SA_scores = np.loadtxt('SA_scores.txt')
	cycle_scores = np.loadtxt('cycle_scores.txt')

	logp_m = np.mean(logP_values)
	logp_s = np.std(logP_values)

	sascore_m = np.mean(SA_scores)
	sascore_s = np.std(SA_scores)

	cycle_m = np.mean(cycle_scores)
	cycle_s = np.std(cycle_scores)
	for i, rxn_tree in enumerate(rxn_trees):
		latent = model.encode([rxn_tree])
		latent_list.append(latent[0])
		smiles = rxn_tree.molecule_nodes[0].smiles
		score_list.append(get_clogp_score(smiles, logp_m, logp_s, sascore_m, sascore_s, cycle_m, cycle_s))
latents = torch.stack(latent_list, dim=0)
scores = np.array(score_list)
scores = scores.reshape((-1,1))
latents = latents.detach().numpy()
n = latents.shape[0]
permutation = np.random.choice(n, n, replace = False)
X_train = latents[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
X_test = latents[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]
y_train = -scores[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
y_test = -scores[ permutation ][ np.int(np.round(0.9 * n)) : ]
print(X_train.shape, X_test.shape)
if metric == "logp":
	parameters = [logp_m, logp_s, sascore_m, sascore_s, cycle_m, cycle_s]
else: 
	parameters =[]

run_bo(X_train, y_train, X_test, y_test, model, parameters, metric, seed)



