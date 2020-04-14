from data import data
import sys
from os import path
import os
from io import open
import string
import numpy as np
import pickle
import linecache
import math
import sys
import tqdm
import torch
from torch import tensor
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from pathlib import Path

class attention_feed(nn.Module):
	
	def __init__(self):
		super(attention_feed, self).__init__()
	
	def forward(self,target_t,context):
		atten=torch.bmm(context,target_t.unsqueeze(2)).sum(2)
		mask=((atten!=0).float()-1)*1e9
		atten=atten+mask
		atten=nn.Softmax(dim=1)(atten)
		atten=atten.unsqueeze(1)
		context_combined=torch.bmm(atten,context).sum(1)
		return context_combined

class softattention(nn.Module):
	
	def __init__(self,params):
		super(softattention, self).__init__()
		dim = params.dimension
		self.attlinear=nn.Linear(dim*2,dim,False)
	
	def forward(self,target_t,context):
		
		atten=torch.bmm(context,target_t.unsqueeze(2)).sum(2)
		mask=((atten!=0).float()-1)*1e9
		atten=atten+mask
		atten=nn.Softmax(dim=1)(atten)
		atten=atten.unsqueeze(1)
		context_combined=torch.bmm(atten,context).sum(1)
		output=self.attlinear(torch.cat((context_combined,target_t),-1))
		output=nn.Tanh()(output)
		return output


class lstm_source(nn.Module):
	
	def __init__(self,params):
		super(lstm_source, self).__init__()
		dim = params.dimension
		layer = params.layers
		self.dropout = nn.Dropout(p=params.dropout)
		self.lstms=nn.LSTM(dim,dim,num_layers=layer,batch_first=True,bias=False,dropout=params.dropout)

	def forward(self,embedding,length):
		embedding = self.dropout(embedding)
		packed=pack_padded_sequence(embedding,length,batch_first=True,enforce_sorted=False)
		
		packed_output,(h,c)=self.lstms(packed)
		context,_= pad_packed_sequence(packed_output,batch_first=True)
		return context,h,c
		

class lstm_target(nn.Module):
	
	def __init__(self,params):
		super(lstm_target, self).__init__()
		
		dim = params.dimension
		persona_dim=params.PersonaDim
		layer = params.layers
		self.dropout = nn.Dropout(p=params.dropout)
		self.speaker = params.SpeakerMode
		self.addressee = params.AddresseeMode
		persona_num = params.PersonaNum
		
		if self.speaker:
			self.persona_embedding=nn.Embedding(persona_num,persona_dim)

			# load emb
			if params.PersonaEmbFiles!='None':
				emb_files=params.PersonaEmbFiles.split(',')
				embs_list=[]
				for emb_file in emb_files:
					embs=self.read_emb_file(emb_file)
					embs_list.append(embs)

				merged_emb=np.concatenate(embs_list,axis=1)
				assert merged_emb.shape[0]==persona_num,(merged_emb.shape[0],persona_num)
				assert merged_emb.shape[1]<=persona_dim,(merged_emb.shape[1],persona_dim)
				rand_init_emb=self.persona_embedding.weight.data.cpu().numpy()
				rand_init_emb[:,0:merged_emb.shape[1]]=merged_emb
				self.persona_embedding.weight.data.copy_(torch.from_numpy(rand_init_emb))
				print('init persona embedding from file',emb_files)
				print('with {} dims'.format(merged_emb.shape[1]))
			else:
				print('randomly init persona embedding')

			if params.debug:
				pass
				# self.persona_embedding.weight.requires_grad = False # TODO:

			self.lstmt=nn.LSTM(dim*2+persona_dim,dim,num_layers=layer,batch_first=True,bias=False,dropout=params.dropout) 
		elif self.addressee:
			self.persona_embedding=nn.Embedding(persona_num,persona_dim)
			self.speaker_linear = nn.Linear(persona_dim,persona_dim)
			self.addressee_linear = nn.Linear(persona_dim,persona_dim)
			self.lstmt=nn.LSTM(dim*2+persona_dim,dim,num_layers=layer,batch_first=True,bias=False,dropout=params.dropout) 
		else:
			self.lstmt=nn.LSTM(dim*2,dim,num_layers=layer,batch_first=True,bias=False,dropout=params.dropout)
		self.atten_feed=attention_feed()
		self.soft_atten=softattention(params)
	def read_emb_file(self,emb_file):
		embs=[]
		with open(emb_file,'r') as fin:
			lines = fin.readlines()
			for i,line in enumerate(lines):
				line = line.strip().split()
				emb=[float(em) for em in line]
				embs.append(emb)
		embs=np.array(embs)
		return embs
		
	def forward(self,context,h,c,embedding,speaker_label,addressee_label):
		embedding = self.dropout(embedding)
		h = self.dropout(h)
		context1=self.atten_feed(h[-1],context)
		context1 = self.dropout(context1)
		lstm_input=torch.cat((embedding,context1),-1)
		if self.speaker:
			speaker_embed=self.persona_embedding(speaker_label)
			speaker_embed = self.dropout(speaker_embed)
			lstm_input=torch.cat((lstm_input,speaker_embed),-1)
		elif self.addressee:
			speaker_embed=self.persona_embedding(speaker_label)
			speaker_embed = self.dropout(speaker_embed)
			addressee_embed=self.persona_embedding(addressee_label)
			addressee_embed = self.dropout(addressee_embed)
			combined_embed = self.speaker_linear(speaker_embed) + self.addressee_linear(addressee_embed)
			combined_embed = nn.Tanh()(combined_embed)
			lstm_input=torch.cat((lstm_input,combined_embed),-1)
		_,(h,c)=self.lstmt(lstm_input.unsqueeze(1),(h,c))
		pred=self.soft_atten(h[-1],context)
		return pred,h,c


class lstm(nn.Module):

	def __init__(self,params,vocab_num,EOT):
		super(lstm, self).__init__()
		dim = params.dimension
		init_weight = params.init_weight
		vocab_num = vocab_num + params.special_word
		print('vocab_num')
		print(vocab_num)
		self.UNK = params.UNK+params.special_word
		self.EOT = EOT
		self.params=params
		
		self.encoder=lstm_source(params)
		self.decoder=lstm_target(params)
		
		self.sembed=nn.Embedding(vocab_num,dim,padding_idx=0)
		self.sembed.weight.data[1:].uniform_(-init_weight,init_weight)
		self.tembed=nn.Embedding(vocab_num,dim,padding_idx=0)
		self.tembed.weight.data[1:].uniform_(-init_weight,init_weight)
		
		self.softlinear=nn.Linear(dim,vocab_num,False)
		w=torch.ones(vocab_num)
		if not params.cpu:
			w = w.cuda()
		w[:self.EOT]=0
		w[self.UNK]=0
		self.loss_function=torch.nn.CrossEntropyLoss(w, ignore_index=0, reduction='sum')

	def forward(self,sources,targets,length,speaker_label,addressee_label):
		source_embed=self.sembed(sources)
		context,h,c=self.encoder(source_embed,length)
		loss=0

		for i in range(targets.size(1)-1):
			target_embed=self.tembed(targets[:,i])
			pred,h,c=self.decoder(context,h,c,target_embed,speaker_label,addressee_label)
			pred=self.softlinear(pred)
			loss+=self.loss_function(pred,targets[:,i+1])
		return loss


class persona:
	
	def __init__(self, params):
		self.best_dev_ppl=np.inf
		self.best_model_cnt=0
		self.params=params
		self.cur_iter=0
		self.cur_epoch=0

		self.ReadDict()
		self.Data=data(params,self.voc)

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if params.cpu:
			self.device="cpu"
		self.Model = lstm(params,len(self.voc),self.Data.EOT)
		self.Model.encoder.apply(self.weights_init)
		self.Model.decoder.apply(self.weights_init)
		self.Model.softlinear.apply(self.weights_init)


		self.Model.to(self.device)
		# make the save_folder
		Path(params.save_folder).mkdir(parents=True, exist_ok=True)

		self.output=path.join(params.save_folder,params.output_file)
		self.output_best_model_log=path.join(params.save_folder,'output_best_model.log')
		
		if self.output!="":
			with open(self.output,"a") as selfoutput:
				selfoutput.write("")
		if self.params.SpeakerMode:
			print("training in speaker mode")
		elif self.params.AddresseeMode:
			print("training in speaker-addressee mode")
		else:
			print("training in non persona mode")

	def weights_init(self,module):
		classname=module.__class__.__name__
		try:
			module.weight.data.uniform_(-self.params.init_weight,self.params.init_weight)
		except:
			pass
	
	def ReadDict(self):
		self.voc = dict()
		with open(path.join(self.params.data_folder,self.params.dictPath),'r') as doc:
			for line in doc:
				self.voc[line.strip()] = len(self.voc)

	def test(self):
		open_train_file=path.join(self.params.data_folder,self.params.dev_file)
		total_loss = 0
		total_tokens = 0
		END=0
		batch_n=0
		while END==0:
			END,sources,targets,speaker_label,addressee_label,length,token_num,origin = self.Data.read_batch(open_train_file,batch_n)
			batch_n+=1
			if sources is None:
				continue
			sources=sources.to(self.device)
			targets=targets.to(self.device)
			speaker_label=speaker_label.to(self.device)
			addressee_label=addressee_label.to(self.device)
			length=length.to(self.device)
			total_tokens+=token_num
			
			self.Model.eval()
			
			with torch.no_grad():
				
				loss = self.Model(sources,targets,length,speaker_label,addressee_label)


				total_loss+=loss.item()
		ppl=1/math.exp(-total_loss/total_tokens)
		print("perp "+str((1/math.exp(-total_loss/total_tokens))))
		if ppl<self.best_dev_ppl:
			self.best_dev_ppl=ppl
			if self.best_dev_ppl<self.params.best_ppl_threshold:
				self.save_best()
		if self.output!="":
			with open(self.output,"a") as selfoutput:
				selfoutput.write("standard perp "+str((1/math.exp(-total_loss/total_tokens)))+"\n")
				

	def update(self):
		lr=self.params.alpha
		grad_norm=0
		for m in list(self.Model.parameters()):
			m.grad.data = m.grad.data*(1/self.source_size)
			grad_norm+=m.grad.data.norm()**2
		grad_norm=grad_norm**0.5
		if grad_norm>self.params.thres:
			lr=lr*self.params.thres/grad_norm
		for f in self.Model.parameters():
			f.data.sub_(f.grad.data * lr)

	def save(self):
		save_path = path.join(self.params.save_folder,self.params.save_prefix)
		torch.save(self.Model.state_dict(),save_path+str(self.iter))
		print("finished saving")

	def save_best(self):
		save_path = path.join(self.params.save_folder,self.params.save_prefix)
		torch.save(self.Model.state_dict(),save_path+'best_{}'.format(int(self.best_model_cnt)))
		
		
		
		with open(self.output_best_model_log,'a') as fout:
			fout.write('best_model_num,{},epoch,{},iter,{:.4},dev_ppl,{}\n'.format(self.best_model_cnt,
				self.cur_epoch,
				self.cur_iter/int(self.params.train_size/self.params.batch_size),
				self.best_dev_ppl))
		self.best_model_cnt+=1
		print("finished saving best model")

	def saveParams(self):
		save_params_path = path.join(self.params.save_folder,self.params.save_params)
		with open(save_params_path,"wb") as file:
			pickle.dump(self.params,file)

	def readModel(self,save_folder,model_name,re_random_weights=None):
		target_model = torch.load(path.join(save_folder,model_name))
		if re_random_weights is not None: 
			for weight_name in re_random_weights:
				random_weight = self.Model.state_dict()[weight_name]
				target_model[weight_name] = random_weight
		self.Model.load_state_dict(target_model)
		print("read model done")

	def train(self):
		if not self.params.no_save:
			self.saveParams()
		if self.params.fine_tuning:
			if self.params.SpeakerMode or self.params.AddresseeMode:
				# re_random_weights = ['decoder.persona_embedding.weight'] # Also have to include some layers of the LSTM module...
				re_random_weights = None # FIXME: we do not re random the weights for now.
				# TODO: use this re_random_weights's method to re-load user emb in combine training. 
			else:
				re_random_weights = None
			print(self.params)
			self.readModel(self.params.save_folder,self.params.fine_tunine_model,re_random_weights)
		self.iter=0
		start_halving=False
		self.lr=self.params.alpha		
		# while True:
		for epoch in tqdm.tqdm(range(self.params.epochs)):
			self.cur_epoch=epoch
			self.iter+=1
			# print("iter  "+str(self.iter))
			if self.output!="":
				with open(self.output,"a") as selfoutput:
					selfoutput.write("iter  "+str(self.iter)+"\n")
			if self.iter>self.params.start_halve:
				self.lr=self.lr*0.5
			open_train_file=path.join(self.params.data_folder,self.params.train_file)
			END=0
			batch_n=0
			for iters in tqdm.tqdm(range(int(self.params.train_size/self.params.batch_size) +1 )):
				self.cur_iter=iters
				
				self.Model.zero_grad()
				END,sources,targets,speaker_label,addressee_label,length,_,_ = self.Data.read_batch(open_train_file,batch_n)


				batch_n+=1
				if sources is None:
					
					continue
					
				sources=sources.to(self.device)
				targets=targets.to(self.device)
				speaker_label=speaker_label.to(self.device)
				addressee_label=addressee_label.to(self.device)
				length=length.to(self.device)
				self.source_size = sources.size(0)
				self.Model.train()
				
				loss = self.Model(sources,targets,length,speaker_label,addressee_label)
				loss.backward()
				self.update()
				if iters%self.params.eval_steps==0:
					print('epoch',epoch,'iter',iters/int(self.params.train_size/self.params.batch_size),' ',end='')
					self.test()


				if END!=0:
					break
			self.test()
			if self.iter%self.params.save_steps==0:
				if not self.params.no_save:
					self.save()
