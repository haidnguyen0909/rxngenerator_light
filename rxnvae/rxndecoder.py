import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem import rdmolops
import torch
import torch.nn as nn
from nnutils import create_var
import math
import torch.nn.functional as F
from rdkit.Chem import rdChemReactions
from collections import deque
from mpn import MPN
from reaction import MoleculeNode, TemplateNode
from reaction_utils import *
from rdkit.Chem import AllChem

def get_template_order(rxn):
	mol_nodes = rxn.molecule_nodes
	tem_nodes = rxn.template_nodes

	order={}
	root = tem_nodes[0]
	queue = deque([root])
	visisted = set([root.id])
	root.depth = 0
	order[0] =[root.id]
	
	while len(queue) > 0:
		x = queue.popleft()
		#print("pop:", x.id)
		for y in x.children:
			if len(y.children) == 0: # starting molecule
				continue
			template = y.children[0] 
			if template.id not in visisted:
				queue.append(template)
				#print("push:", template.id)
				visisted.add(template.id)
				template.depth = x.depth + 1
				if template.depth not in order:
					order[template.depth] = [template.id]
				else:
					order[template.depth].append(template.id)
	return order

class RXNDecoder(nn.Module):
	def __init__(self, hidden_size, latent_size, reactantDic, templateDic, molecule_embedding=None, template_embedding=None, mpn = None):
		super(RXNDecoder, self).__init__()
		self.hidden_size = hidden_size
		self.latent_size = latent_size
		self.reactantDic = reactantDic
		self.templateDic = templateDic

		if template_embedding is None:
			self.template_embedding = nn.Embedding(self.templateDic.size(), self.hidden_size)
		else:
			self.template_embedding = template_embedding

		if molecule_embedding is None:
			self.molecule_embedding = nn.Embedding(self.reactantDic.size(), self.hidden_size)
		else:
			self.molecule_embedding = molecule_embedding
		if mpn is None:
			self.mpn = MPN(self.hidden_size, 2)
		else:
			self.mpn = mpn

		self.molecule_distance_loss = nn.MSELoss(size_average = False)
		self.template_loss = nn.CrossEntropyLoss(size_average = False)
		self.molecule_label_loss = nn.CrossEntropyLoss(size_average = False) 

		self.W_root = nn.Linear(self.latent_size, self.hidden_size)
		self.W_template = nn.Linear(self.hidden_size + self.latent_size, self.templateDic.size())
		self.W_reactant_out = nn.Linear(self.hidden_size + self.latent_size, self.hidden_size)
		self.W_label = nn.Linear(self.hidden_size, self.reactantDic.size())



		# update molecules
		self.gru = nn.GRU(2 * self.hidden_size + self.latent_size, self.hidden_size)
		self.gru_template = nn.GRU(self.latent_size, self.hidden_size)


	def decode_many_time(self, latent_vector, n):
		results =[]
		for i in range(n):
			res1, res2 = self.decode(latent_vector)
			if res1 != None:
				return res1, res2
		return None, None


	def decode(self, latent_vector, prob_decode=True):
		#context_vector = attention(encoder_outputs, latent_vector)
		root_embedding = self.W_root(torch.cat([latent_vector], dim=1))
		root_embedding = nn.ReLU()(root_embedding)

		molecule_labels ={}
		template_labels ={}
		molecule_hs ={}
		template_hs ={}
		
		queue = deque([])
		molecule_nodes =[]
		template_nodes =[]
		molecule_counter = 0
		template_counter = 0
		tree_root = MoleculeNode("", molecule_counter)
		molecule_hs[molecule_counter] = root_embedding
		molecule_nodes.append(tree_root)
		molecule_labels[molecule_counter] = self.reactantDic.get_index("unknown")
		molecule_counter += 1


		# predict template
		product_vector = molecule_hs[0]
		#context = attention(encoder_outputs, product_vector)
		prev_xs = torch.cat([latent_vector], dim=1).unsqueeze(0)
		os, hs = self.gru_template(prev_xs, product_vector.unsqueeze(0))
		hs = hs[0,:,:]

		#context = attention(encoder_outputs, hs)
		logits = self.W_template(torch.cat([latent_vector, hs], dim=1))
		output = F.softmax(logits, dim=1)
		if prob_decode:
			template_type = torch.multinomial(output[0], 1)
		else:
			_, template_type = torch.max(output, dim=1)
		template_node = TemplateNode("", template_counter)
		template_node.template_type = template_type
		template_node.parents.append(tree_root)
		tree_root.children.append(template_node)
		template_nodes.append(template_node)
		template_hs[template_counter] = hs
		template_labels[template_counter] = template_type
		template_counter += 1

		n_reactants = self.templateDic.get_n_reacts(template_type.item())
		product_id = template_node.parents[0].id
		for n in range(n_reactants):
			if n == 0:
				temp_id = create_var(torch.LongTensor(template_node.template_type))
				pre_xs = self.template_embedding(temp_id)
				pre_hs = template_hs[template_node.id]
			else:
				prev_mol_id = template_node.children[n-1].id
				mol_label =molecule_labels[prev_mol_id]
				#print(mol_label)
				pre_xs = self.molecule_embedding(mol_label)
				pre_hs = molecule_hs[prev_mol_id]
			#context = attention(encoder_outputs, pre_hs)
			pre_xs = torch.cat([latent_vector, pre_xs, molecule_hs[product_id]], dim=1)
			os, hs = self.gru(pre_xs.unsqueeze(0), pre_hs.unsqueeze(0))
			hs = hs[0,:,:]
			molecule_hs[molecule_counter] = hs
			#context = attention(encoder_outputs, hs)
			input = torch.cat([latent_vector, hs], dim = 1)
			output = nn.ReLU()(self.W_reactant_out(input))
			output = self.W_label(output)
			output = F.softmax(output, dim=1)
			#label = output.max(dim=-1)
			
			if prob_decode:
				label = torch.multinomial(output[0], 1)
			else:
				_, label = torch.max(output, dim=1)
			molecule_labels[molecule_counter] = label
			reactant_node = MoleculeNode("", molecule_counter)
			reactant_node.parents.append(template_node)
			template_node.children.append(reactant_node)
			queue.append(reactant_node)
			molecule_counter+=1
		count =1
		while len(queue)> 0:
			#print(queue)
			count +=1
			if count > 20:
				return None, None
			cur_molecule_node = queue.popleft()
			molecule_nodes.append(cur_molecule_node)
			template_node = cur_molecule_node.parents[0]
			pre_molecule_node = template_node.parents[0]

			temp_id = template_node.id
			pre_molecule_id = pre_molecule_node.id

			cur_molecule_vec = molecule_hs[cur_molecule_node.id]
			pre_molecule_vec = molecule_hs[pre_molecule_node.id]
			template_vec = template_hs[template_node.id]

			input = torch.cat([latent_vector, cur_molecule_vec], dim=1)
			output = nn.ReLU()(self.W_reactant_out(input))
			output = self.W_label(output)
			output = F.softmax(output, dim=1)
			if prob_decode:
				output = torch.multinomial(output[0], 1)
			else:
				_, output = torch.max(output, dim=1)
			if output.item()== self.reactantDic.size()-1:
				pre_xs  = torch.cat([latent_vector], dim=1).unsqueeze(0)
				os, hs = self.gru_template(pre_xs, cur_molecule_vec.unsqueeze(0))
				hs = hs[0,:,:]

				#context = attention(encoder_outputs, hs)
				logits = self.W_template(torch.cat([latent_vector, hs], dim=1))
				output = F.softmax(logits, dim=1)
				#template_type = torch.multinomial(output[0], 1)
				_, template_type = torch.max(output, dim=1)
				template_node = TemplateNode("", template_counter)
				template_node.parents.append(cur_molecule_node)
				template_node.template_type = template_type
				cur_molecule_node.children.append(template_node)
				template_nodes.append(template_node)
				template_labels[template_counter] = template_type
				template_hs[template_counter] = hs
				n_reactants = self.templateDic.get_n_reacts(template_type.item())

				for n in range(n_reactants):
					if n==0:
						temp_id = create_var(torch.LongTensor(template_node.template_type))
						pre_xs = self.template_embedding(temp_id)
						pre_hs = template_hs[template_node.id]
					else:
						prev_mol_id = template_node.children[n-1].id
						mol_label =molecule_labels[prev_mol_id]
						#print(mol_label)
						pre_xs = self.molecule_embedding(mol_label)
						pre_hs = molecule_hs[prev_mol_id]
					pre_xs = torch.cat([latent_vector, pre_xs, cur_molecule_vec], dim=1)
					os, hs = self.gru(pre_xs.unsqueeze(0), pre_hs.unsqueeze(0))
					hs = hs[0,:,:]
					molecule_hs[molecule_counter] = hs
					input = torch.cat([latent_vector, hs], dim = 1)
					output = nn.ReLU()(self.W_reactant_out(input))
					output = self.W_label(output)
					output = F.softmax(output, dim=1)
					
					if prob_decode:
						label = torch.multinomial(output[0], 1)
					else:
						_, label = torch.max(output, dim=1)
					molecule_labels[molecule_counter] = label
					reactant_node = MoleculeNode("", molecule_counter)
					reactant_node.parents.append(template_node)
					template_node.children.append(reactant_node)
					#print(n, molecule_labels)

					queue.append(reactant_node)
					molecule_counter+=1
				template_counter +=1
			else:
				cur_molecule_node.reactant_id = output[0].item()



		node2smiles ={}
		root = template_nodes[0]
		queue = deque([root])
		visited = set([root.id])
		root.depth = 0
		order ={}
		order[0] = [root.id]
		while len(queue) > 0:
			x = queue.popleft()
			for y in x.children:
				if len(y.children) == 0:
					continue
				template = y.children[0]
				if template.id not in visited:
					queue.append(template)
					visited.add(template.id)
					template.depth = x.depth + 1
					if template.depth not in order:
						order[template.depth] = [template.id]
					else:
						order[template.depth].append(template.id)
		max_depth = len(order) - 1
		for t in range(max_depth, -1, -1):
			for template_id in order[t]:
				template_node = template_nodes[template_id]
				template_type = template_node.template_type.item()
				template = self.templateDic.get_template(template_type)
				reactants = []
				for reactant_node in template_node.children:
					if len(reactant_node.children) == 0:
						reactant = self.reactantDic.get_reactant(reactant_node.reactant_id)
						reactants.append(reactant)
						node2smiles[reactant_node.id] = reactant
					else:
						reactant = node2smiles[reactant_node.id]
						reactants.append(reactant)
				possible_templates = reverse_template(template)
				#print("before:", template)
				#parts = template.split(">>")
				#template = ">>".join([parts[1], parts[0]])
				#print("after:", template)
				possible_products = []
				reacts = [Chem.MolFromSmiles(reactant) for reactant in reactants]
				#react_combs = list(itertools.permutations(reacts))
				for possible_template in possible_templates:
					#print(reactants, template)
					try:
						rxn = rdChemReactions.ReactionFromSmarts(possible_template)
						AllChem.SanitizeRxn(rxn)
						#for reacts in possible_reacts:
						products = rxn.RunReactants(reacts)
						#print(template,possible_templates, reactants)
						#print(reactants, self.templateDic.get_n_reacts(self.templateDic.get_index(template)),len(possible_templates), template)
					except:
						continue

					if len(products) > 0:
						n = len(products)
						for i in range(n):
							product = products[i]
							possible_products.append(product[0])
				if len(possible_products) > 0:
					product_id = template_node.parents[0].id
					node2smiles[product_id] = Chem.MolToSmiles(possible_products[0])
				else:
					success = False
					return None, None
		str_reactions=[]
		for t in range(len(order)):
			for template_id in order[t]:
				template_node = template_nodes[template_id]
				template_type = template_node.template_type.item()
				template = self.templateDic.get_template(template_type)
				reactants =[]
				for reactant_node in template_node.children:
					reactant = node2smiles[reactant_node.id]
					reactants.append(reactant)
				product_id = template_node.parents[0].id
				product = node2smiles[product_id]
				reactants =".".join(reactants)
				reaction = "$".join([product, reactants, template])
				str_reactions.append(reaction)
		str_reactions = " ".join(str_reactions)
		return node2smiles[0], str_reactions


	def forward(self, rxn_tree_batch, latent_vectors):
		# intiualize sth
		batch_size = len(rxn_tree_batch)
		
		template_acc = 0
		n_templates = 0
		molecule_distance_loss = 0
		n_molecules = 0
		
		template_loss = 0
		molecule_label_loss = 0

		label_acc = 0

		orders =[]
		B = len(rxn_tree_batch)
		for rxn_tree in rxn_tree_batch:
			order = get_template_order(rxn_tree)
			orders.append(order)
		max_depth = max([len(order) for order in orders])

		target_molecules =[]
		target_templates =[]
		molecule_labels={}

		for rxn_id in range(len(rxn_tree_batch)):
			molecule_nodes = rxn_tree_batch[rxn_id].molecule_nodes
			template_nodes = rxn_tree_batch[rxn_id].template_nodes
			for ind, molecule_node in enumerate(molecule_nodes):
				#target_molecules.append(molecule_node.smiles)
				if len(molecule_node.children)==0:
					molecule_labels[(rxn_id, ind)] = self.reactantDic.get_index(molecule_node.smiles)
				else:
					molecule_labels[(rxn_id, ind)] = self.reactantDic.get_index("unknown")
			for template_node in template_nodes:
				target_templates.append(self.templateDic.get_index(template_node.template))
		#target_mol_embeddings = self.mpn(target_molecules)

		
		o_target ={}
		l_target ={}
		h_pred ={}
		l_pred ={}
		o_pred = {}
		logits_pred={}
		template_hids={}
		template_outs={}
		i = 0
		for rxn_id in range(len(rxn_tree_batch)):
			for j in range(len(rxn_tree_batch[rxn_id].template_nodes)):
				l_target[(rxn_id, j)] = target_templates[i]
				i+=1

		#context_vectors = attention(encoder_outputs, latent_vectors)
		root_embeddings = self.W_root(torch.cat([latent_vectors], dim=1))
		root_embeddings = nn.ReLU()(root_embeddings)

		

		for rxn_id in range(B):
			h_pred[(rxn_id, 0)] = root_embeddings[rxn_id]
			o_pred[(rxn_id, 0)]= root_embeddings[rxn_id]

		for t in range(max_depth):
			tem_E ={}
			template_targets =[]
			template_hs ={}
			# for tracking
			template_ids =[]
			rxn_ids =[]
			template2reactants={}
			for i, order in enumerate(orders):
				if len(order) > t:
					template_ids.extend(order[t])
					rxn_ids.extend([i]*len(order[t]))
			product_vectors_t =[]
			latent_vectors_t =[]
			n_reactants =[]
			tem_targets =[]
			for template_id, rxn_id in zip(template_ids, rxn_ids):
				template_node = rxn_tree_batch[rxn_id].template_nodes[template_id]
				product_id = template_node.parents[0].id
				product_vectors_t.append(h_pred[(rxn_id, product_id)])
				latent_vectors_t.append(latent_vectors[rxn_id])
				tem_targets.append(self.templateDic.get_index(template_node.template))
				#template2reactants.append(len(template_node.children))
				n_reactants.append(len(template_node.children))
	
			product_vectors_t = torch.stack(product_vectors_t, dim=0)
			latent_vectors_t = torch.stack(latent_vectors_t, dim=0)
			#context = attention(cur_enc_outputs, product_vectors_t)
			prev_xs = torch.cat([latent_vectors_t], dim=1).unsqueeze(0) 
			os, hs = self.gru_template(prev_xs, product_vectors_t.unsqueeze(0))
			hs = hs[0,:,:]
			i=0
			for template_id, rxn_id in zip(template_ids, rxn_ids):
				template_hs[(rxn_id, template_id)] = hs[i]
				i+=1

			logits = self.W_template(torch.cat([latent_vectors_t, hs], dim=1))
			
			# for target
			tem_vecs = self.template_embedding(create_var(torch.LongTensor(tem_targets)))
			i=0
			
			for template_id, rxn_id in zip(template_ids, rxn_ids):
				tem_E[(rxn_id, template_id)] = tem_vecs[i]
				i+=1

			i=0
			for template_id, rxn_id in zip(template_ids, rxn_ids):
				logits_pred[(rxn_id, template_id)] = logits[i]
				i+=1


			tem_targets = create_var(torch.LongTensor(tem_targets))
			tloss = self.template_loss(logits, tem_targets)
			template_loss += tloss
			#i=0
			#for template_id, rxn_id in zip(template_ids, rxn_ids):
			#	template_loss[rxn_id] += tloss[i]
			#	i+=1
			
			output = F.softmax(logits, dim=1)
			#print(output)
			_, output = output.max(dim=-1)
			template_acc += (output == tem_targets).float().sum()
			n_templates += tem_targets.shape[0]

			i = 0
			max_n_reactants = max(n_reactants)


			for n in range(max_n_reactants):
				rxn_tem_mols= []
				prev_hs = []
				prev_xs = []
				latent_vectors_t = []
				product_vectors_t = []
				template_vectors_t = []
				encoder_outputs_t =[]
				mol_targets=[]
				target_labels=[]
				mol_ids =[]
				mol_E={}

				for template_id, rxn_id in zip(template_ids, rxn_ids):
					template_node = rxn_tree_batch[rxn_id].template_nodes[template_id]
					if len(template_node.children) > n:
						if n == 0:
							continue
						else:
							#prev_mol_id = template_node.children[n-1].smiles
							molecule_node = template_node.children[n-1]
							if len(molecule_node.children) == 0:
								id = self.reactantDic.get_index(template_node.children[n-1].smiles)
							else:
								id = self.reactantDic.get_index("unknown")
							mol_ids.append(id)

				mol_ids = create_var(torch.LongTensor(mol_ids))
				embeddings = self.molecule_embedding(mol_ids)
				i=0
				
				for template_id, rxn_id in zip(template_ids, rxn_ids):
					template_node = rxn_tree_batch[rxn_id].template_nodes[template_id]
					if len(template_node.children) > n:
						if n == 0:
							continue
						else:
							prev_mol_id = template_node.children[n-1].id
							mol_E[(rxn_id, prev_mol_id)] = embeddings[i]
							i+=1
				
				for template_id, rxn_id in zip(template_ids, rxn_ids):
					template_node = rxn_tree_batch[rxn_id].template_nodes[template_id]
					if len(template_node.children) > n:
						mol_id = template_node.children[n].id
						mol_targets.append(template_node.children[n].smiles)
						product_id = template_node.parents[0].id
						rxn_tem_mols.append((rxn_id, template_id, mol_id))
						target_labels.append(molecule_labels[(rxn_id, mol_id)])

						latent_vectors_t.append(latent_vectors[rxn_id])
						product_vectors_t.append(h_pred[(rxn_id, product_id)])
						template_vectors_t.append(l_target[(rxn_id, template_id)])

						#mol_targets.append(o_target[(rxn_id, mol_id)])
						if n==0:
							prev_xs.append(tem_E[(rxn_id, template_id)])
							prev_hs.append(template_hs[(rxn_id, template_id)])
						else:
							prev_mol_id = template_node.children[n-1].id
							prev_xs.append(mol_E[(rxn_id, prev_mol_id)])
							prev_hs.append(h_pred[(rxn_id, prev_mol_id)])


				prev_hs = torch.stack(prev_hs, dim=0)
				prev_xs = torch.stack(prev_xs, dim=0)
				latent_vectors_t = torch.stack(latent_vectors_t, dim = 0)
				product_vectors_t = torch.stack(product_vectors_t, dim=0)
				#template_vectors_t = self.template_embedding(create_var(torch.LongTensor(template_vectors_t)))
				#context = attention(encoder_outputs_t, prev_hs)
				prev_xs = torch.cat([latent_vectors_t, prev_xs, product_vectors_t], dim=1)

				os, hs = self.gru(prev_xs.unsqueeze(0), prev_hs.unsqueeze(0))# pre_hs-> product
				hs, os = hs[0,:,:], os[0,:,:]
				for i in range(len(rxn_tem_mols)):
					rxn_id, template_id, mol_id = rxn_tem_mols[i]
					h_pred[(rxn_id, mol_id)] = hs[i]
					o_pred[(rxn_id, mol_id)] = os[i]
				#context = attention(encoder_outputs_t, hs)
				mol_preds = nn.ReLU()(self.W_reactant_out(torch.cat([latent_vectors_t, hs], dim=1)))
				#mol_targets = self.mpn(mol_targets)
				#molecule_distance_loss += self.molecule_distance_loss(mol_preds, mol_targets)
				n_molecules += mol_preds.shape[0]

				pred_labels = self.W_label(mol_preds)
				target_labels = create_var(torch.LongTensor(target_labels))
				mloss = self.molecule_label_loss(pred_labels, target_labels)

				molecule_label_loss += mloss

				pred_labels = F.softmax(pred_labels, dim=1)
				_, pred_labels = pred_labels.max(dim=-1)
				label_acc += (pred_labels==target_labels).float().sum()

		return 0.1 * molecule_distance_loss, template_loss, molecule_label_loss, template_acc/n_templates, label_acc/n_molecules




























