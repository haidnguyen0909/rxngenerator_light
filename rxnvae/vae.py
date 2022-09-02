
import torch
import torch.nn as nn
from nnutils import create_var, attention
from rxndecoder import RXNDecoder
from rxnencoder import RXNEncoder
from mpn import MPN,PP,Discriminator


def set_batch_nodeID(ft_trees, ft_vocab):
	tot = 0
	for ft_tree in ft_trees:
		for node in ft_tree.nodes:
			node.idx = tot
			node.wid = ft_vocab.get_index(node.smiles)
			tot +=1
def log_Normal_diag(x, mean, log_var):
	log_normal = -0.5 * ( log_var + torch.pow( x - mean, 2 ) / torch.exp( log_var ) )
	return torch.mean(log_normal)


class RXNVAE(nn.Module):
	def __init__(self, reactant_vocab, template_vocab, hidden_size, latent_size, depth, reactant_embedding=None, template_embedding=None):
		super(RXNVAE, self).__init__()
		self.reactant_vocab = reactant_vocab
		self.template_vocab = template_vocab
		self.depth = depth

		self.hidden_size = hidden_size
		self.latent_size = latent_size


		if reactant_embedding is None:
			self.reactant_embedding = nn.Embedding(self.reactant_vocab.size(), hidden_size)
		else:
			self.reactant_embedding = reactant_embedding

		if template_embedding is None:
			self.template_embedding = nn.Embedding(self.template_vocab.size(), hidden_size)
		else:
			self.template_embedding = template_embedding
		self.mpn = MPN(hidden_size, 2)

		self.rxn_decoder = RXNDecoder(self.hidden_size, self.latent_size, self.reactant_vocab, self.template_vocab, self.reactant_embedding, self.template_embedding, self.mpn)
		self.rxn_encoder = RXNEncoder(self.hidden_size, self.latent_size, self.reactant_vocab, self.template_vocab, self.mpn, self.template_embedding)

		self.RXN_mean = nn.Linear(hidden_size, latent_size)
		self.RXN_var = nn.Linear(hidden_size, latent_size)

	def encode(self, rxn_trees):
		batch_size = len(rxn_trees)
		root_vecs_rxn = self.rxn_encoder(rxn_trees)
		rxn_mean = self.RXN_mean(root_vecs_rxn)
		return rxn_mean
	def forward(self, rxn_trees, beta, epsilon_std=1.0):
		batch_size = len(rxn_trees)
		root_vecs_rxn = self.rxn_encoder(rxn_trees)
		rxn_mean = self.RXN_mean(root_vecs_rxn)
		rxn_log_var = -torch.abs(self.RXN_var(root_vecs_rxn))

		epsilon = create_var(torch.randn(batch_size, int(self.latent_size)), False)*epsilon_std
		rxn_vec = rxn_mean + torch.exp(rxn_log_var / 2) * epsilon
		molecule_distance_loss, template_loss, molecule_loss, template_acc, molecule_acc = self.rxn_decoder(rxn_trees, rxn_vec)

		kl_loss = -0.5 * torch.sum(1.0 + rxn_log_var - rxn_mean * rxn_mean - torch.exp(rxn_log_var)) / batch_size

		total_loss = template_loss + molecule_loss + beta * kl_loss

		return total_loss, template_loss, molecule_loss, template_acc, molecule_acc, kl_loss
		



















































