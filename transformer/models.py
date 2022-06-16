import fastNLP
import torch
import math
from fastNLP.modules.encoder.transformer import TransformerEncoder
from fastNLP.modules.decoder.crf import ConditionalRandomField
from fastNLP import Const
import copy
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import transformer

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class Embedding(nn.Module):
    def __init__(self,task_size, d_model, word_embedding=None, bi_embedding=None, word_size=None,bi_size=None ,freeze=True):
        super(Embedding, self).__init__()
        self.task_size=task_size        
        self.embed_dim = 0        
        
        self.task_embed = nn.Embedding(task_size,d_model)
        """
        if freeze:
            self.task_embed.weight.requires_grad = False
        """
        if word_embedding is not None:
            self.uni_embed = nn.Embedding.from_pretrained(torch.FloatTensor(word_embedding), freeze=freeze)
            self.embed_dim+=word_embedding.shape[1]
        else:
            # 不使用预训练的word_embdedding
            if bi_embedding is not None:
                # 存在bi_embedding的话使用它
                self.embed_dim+=bi_embedding.shape[1]
            else: self.embed_dim=d_model
            assert word_size is not None
            # 使用随机初始化的uni_embed
            self.uni_embed = nn.Embedding(word_size,self.embed_dim)

        # 如果没有word_embedding，那就随机初始化一个，接下来是处理bi_embedding的情况
        # 这里没有考虑没有bi_embedding的情况，试一下随机初始化bi_embedding的情况
        if bi_embedding is not None:    
            self.bi_embed = nn.Embedding.from_pretrained(torch.FloatTensor(bi_embedding), freeze=freeze)
            self.embed_dim += bi_embedding.shape[1]*2
        # 没有bi_embedding直接不使用
        # else:
        #     # 使用随机初始化的bi_embedding
        #     self.bi_embed = nn.Embedding(bi_size,self.embed_dim)
        #     self.embed_dim += self.bi_embed.shape[1]*2
            
        print("Trans Freeze",freeze,self.embed_dim)
        
        if d_model!=self.embed_dim:
            self.F=nn.Linear(self.embed_dim,d_model)
        else :
            self.F=None

        self.d_model = d_model

    def forward(self, task, uni, bi1=None, bi2=None):
        #print(task,uni.size(),bi1.size(),bi2.size())
        #print(bi1,bi2)
        #assert False
        # uni_embed 是 (word_size,d_model)
        # bi_embed 是 (bi_size,d_model)
        # y 是 (x, d_model)
        y_task=self.task_embed(task[:,0:1])
        y=self.uni_embed(uni[:,1:])
        if bi1 is not None:
            assert self.bi_embed is not None
            
            y=torch.cat([y,self.bi_embed(bi1),self.bi_embed(bi2)],dim=-1)
            #y2=self.bi_embed(bi)
            #y=torch.cat([y,y2[:,:-1,:],y2[:,1:,:]],dim=-1)
            
        #y=torch.cat([y_task,y],dim=1)

        # F是nn.Linear(self.embed_dim,d_model) 其实 是从 3 * d_model 到 d_model的映射
        # 如果bi1和bi2都是None的话，
        if self.F is not None:
            y=self.F(y)
        y=torch.cat([y_task,y],dim=1)
        return y * math.sqrt(self.d_model)

def seq_len_to_mask(seq_len,max_len=None):
    if isinstance(seq_len, np.ndarray):
        assert len(np.shape(seq_len)) == 1, f"seq_len can only have one dimension, got {len(np.shape(seq_len))}."
        if max_len is None:
            max_len = int(seq_len.max())
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)
    
    elif isinstance(seq_len, torch.Tensor):
        assert seq_len.dim() == 1, f"seq_len can only have one dimension, got {seq_len.dim() == 1}."
        batch_size = seq_len.size(0)
        if max_len is None:
            max_len = seq_len.max().long()
        broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
        mask = broad_cast_seq_len.lt(seq_len.unsqueeze(1))
    else:
        raise TypeError("Only support 1-d numpy.ndarray or 1-d torch.Tensor.")
    
    return mask        
    
class CWSModel(nn.Module):
    def __init__(self, encoder, src_embed, position, d_model, tag_size, crf=None):
        super(CWSModel, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.pos=copy.deepcopy(position)
        self.proj = nn.Linear(d_model, tag_size)
        self.tag_size=tag_size
        if crf is None:
            self.crf=None
            self.loss_f=nn.CrossEntropyLoss(size_average=False)
        else:
            print("crf")
            trans=fastNLP.modules.decoder.crf.allowed_transitions(crf,encoding_type='bmes')
            self.crf=ConditionalRandomField(tag_size,allowed_transitions=trans)
        #self.norm=nn.LayerNorm(d_model)

    def forward(self, task, uni, seq_len, bi1=None, bi2=None, tags=None):
        # 没有预训练的uni_embedding和bi_embedding时，在main之中通过model(batch_x["task"],batch_x["uni"],batch_x["seq_len"],tags = batch_y["tags"])调用
        #mask=fastNLP.core.utils.seq_len_to_mask(seq_len)
        mask=seq_len_to_mask(seq_len,uni.size(1))
        out=self.src_embed(task,uni,bi1,bi2)
        out=self.pos(out)
        #out=self.norm(out)
        #print(uni.size(),out.size(),mask.size(),seq_len)
        out=self.proj(self.encoder(out, mask.float()))
        
        if self.crf is not None:
            if tags is not None:
                out=self.crf(out, tags, mask)
                return {"loss":out}
            else: 
                out,_ =self.crf.viterbi_decode(out, mask)
                return {"pred":out}
        else:
            if tags is not None:
                num=out.size(0)
                loss = self.loss_f(torch.masked_select(out,mask.unsqueeze(-1).expand_as(out)).contiguous().view(-1,self.tag_size), torch.masked_select(tags,mask))
                return {"loss":loss/num}
            else:
                out=torch.argmax(out,dim=-1)  
                return {"pred":out}
   
                
def make_CWS(N=6, d_model=256, d_ff=1024, h=4, dropout=0.2, tag_size=4, task_size=8, bigram_embedding=None, word_embedding=None, word_size=None, bi_size=None, crf=None,freeze=True):
    c = copy.deepcopy
    #encoder=TransformerEncoder(num_layers=N,model_size=d_model,inner_size=d_ff,key_size=d_model//h,value_size=d_model//h,num_head=h,dropout=dropout)
    encoder=transformer.make_encoder(N=N,d_model=d_model,h=h,dropout=dropout,d_ff=d_ff)
    
    position = PositionalEncoding(d_model, dropout)
    embed=Embedding(task_size, d_model, word_embedding, bigram_embedding,word_size,bi_size, freeze)

    model=CWSModel(encoder, embed, position, d_model, tag_size, crf=crf)

    
    for name,p in model.named_parameters():
        if p.dim() > 1 and p.requires_grad==True:
            nn.init.xavier_uniform(p)
    return model
