import argparse
import pickle
import collections
import logging
import math
import os,sys,time
import random
from sys import maxsize
import pickle
import numpy as np
import torch
import torch.nn as nn
import fastNLP
from fastNLP import BucketSampler,SequentialSampler
from fastNLP import DataSetIter
import optm
import models
import utils


NONE_TAG = "<NONE>"
START_TAG = "<sos>"
END_TAG = "<eos>"

DEFAULT_WORD_EMBEDDING_SIZE = 100
DEBUG_SCALE = 200

# ===-----------------------------------------------------------------------===
# Argument parsing
# ===-----------------------------------------------------------------------===
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, dest="dataset", help=".pkl file to use")
parser.add_argument("--dict", dest="dict", required=True, help=".pkl file to use")
parser.add_argument("--word-embeddings", dest="word_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--bigram-embeddings", dest="bigram_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--crf", dest="crf", action="store_true", help="whether to use CRF")                    
parser.add_argument("--devi", default="0", dest="devi", help="gpu to use")
parser.add_argument("--step", default=0, dest="step", type=int,help="step")
parser.add_argument("--num-epochs", default=80, dest="num_epochs", type=int,
                    help="Number of epochs through training set")
parser.add_argument("--flex", default=-1, dest="flex", type=int,
                    help="Number of epochs through training set after freezing the pretrained embeddings")
parser.add_argument("--batch-size", default=256, dest="batch_size", type=int,
                    help="Minibatch size of training set")
parser.add_argument("--d_model", default=256, dest="d_model", type=int, help="d_model of transformer encoder")
parser.add_argument("--d_ff", default=1024, dest="d_ff", type=int, help="d_ff for FFN")
parser.add_argument("--N", default=6, dest="N", type=int, help="Number of layers")
parser.add_argument("--h", default=4, dest="h", type=int, help="Number of head")
parser.add_argument("--factor", default=2, dest="factor", type=float, help="factor for learning rate")
parser.add_argument("--dropout", default=0.2, dest="dropout", type=float,
                    help="Amount of dropout(not keep rate, but drop rate) to apply to embeddings part of graph")
parser.add_argument("--log-dir", default="result", dest="log_dir",
                    help="Directory where to write logs / saved models")
parser.add_argument("--task-name", default=time.strftime("%Y-%m-%d-%H-%M-%S"), dest="task_name",
                    help="Name for this task, use a comprehensive one")
parser.add_argument("--no-model", dest="no_model", action="store_true", help="Don't save model")
parser.add_argument("--always-model", dest="always_model", action="store_true",
                    help="Always save the model after every epoch")
parser.add_argument("--old-model", dest="old_model", help="Path to old model for incremental training")
parser.add_argument("--skip-dev", dest="skip_dev", action="store_true", help="Skip dev set during training")
parser.add_argument("--freeze", dest="freeze", action="store_true", help="freeze pretrained embeddings")
parser.add_argument("--only-task", dest="only_task", action="store_true", help="only train task embeddings")
parser.add_argument("--subset", dest="subset", help="Only train and test on a subset of the whole dataset")
parser.add_argument("--seclude", dest="seclude", help="train and test except a subset of the copora")
parser.add_argument("--instances", default=None, dest="instances", type=int,help="num of instances of subset")

parser.add_argument("--python-seed", dest="python_seed", type=int, default=random.randrange(maxsize),
                    help="Random seed of Python and NumPy")
parser.add_argument("--debug", dest="debug", default=False, action="store_true", help="Debug mode")
parser.add_argument("--test", dest="test", action="store_true", help="Test mode")



options = parser.parse_args()
task_name = options.task_name
root_dir = "{}/{}".format(options.log_dir, task_name)
utils.make_sure_path_exists(root_dir)

devices=[int(x) for x in options.devi]
device = torch.device("cuda:{}".format(devices[0]))  

def init_logger():
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    log_formatter = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    file_handler = logging.FileHandler("{0}/info.log".format(root_dir), mode='w')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


# ===-----------------------------------------------------------------------===
# Set up logging
# ===-----------------------------------------------------------------------===
logger = init_logger()

# ===-----------------------------------------------------------------------===
# Log some stuff about this run
# ===-----------------------------------------------------------------------===
logger.info(' '.join(sys.argv))
logger.info('')
logger.info(options)

if options.debug:
    print("DEBUG MODE")
    options.num_epochs = 2
    options.batch_size=20

random.seed(options.python_seed)
np.random.seed(options.python_seed % (2 ** 32 - 1))
logger.info('Python random seed: {}'.format(options.python_seed))

# ===-----------------------------------------------------------------------===
# Read in dataset
# ===-----------------------------------------------------------------------===
dataset = pickle.load(open(options.dataset, "rb"))
train_set=dataset["train_set"]
test_set=dataset["test_set"]
uni_vocab=dataset["uni_vocab"]
bi_vocab=dataset["bi_vocab"]
task_vocab=dataset["task_vocab"]
tag_vocab=dataset["tag_vocab"]
print(bi_vocab.to_word(0),tag_vocab.word2idx)
print(task_vocab.word2idx)
if options.skip_dev:
    dev_set=test_set
else:
    train_set, dev_set=train_set.split(0.1)
    
print(len(train_set),len(dev_set),len(test_set))

if options.debug:
    train_set = train_set[0:DEBUG_SCALE]
    dev_set = dev_set[0:DEBUG_SCALE]
    test_set = test_set[0:DEBUG_SCALE]

# ===-----------------------------------------------------------------------===
# Build model and trainer
# ===-----------------------------------------------------------------------===

if options.word_embeddings is None:
    init_embedding=None
else:
    print("Load:",options.word_embeddings)
    init_embedding=fastNLP.io.embed_loader.EmbedLoader.load_with_vocab(options.word_embeddings, uni_vocab, normalize=False)
    
bigram_embedding = None
if options.bigram_embeddings:
    if options.bigram_embeddings == 'merged':
        logging.info('calculate bigram embeddings from unigram embeddings')
        bigram_embedding=np.random.randn(len(bi_vocab), init_embedding.shape[-1]).astype('float32')      
        for token, i in bi_vocab:
            if token.startswith('<') and token.endswith('>'): continue
            if token.endswith('>'):
                x,y=uni_vocab[token[0]], uni_vocab[token[1:]]
            else: 
                x,y=uni_vocab[token[:-1]], uni_vocab[token[-1]]
            if x==uni_vocab['<unk>']:
                x=uni_vocab['<pad>']
            if y==uni_vocab['<unk>']:
                y=uni_vocab['<pad>']
            bigram_embedding[i]=(init_embedding[x]+init_embedding[y])/2
    else:    
        print("Load:",options.bigram_embeddings)
        bigram_embedding=fastNLP.io.embed_loader.EmbedLoader.load_with_vocab(options.bigram_embeddings, bi_vocab, normalize=False)

#select subset training
if options.seclude is not None:
    setname="<{}>".format(options.seclude)
    print("seclude",setname)
    train_set.drop(lambda x: x["ori_words"][0]==setname,inplace=True)
    test_set.drop(lambda x: x["ori_words"][0]==setname,inplace=True)
    dev_set.drop(lambda x: x["ori_words"][0]==setname,inplace=True)

if options.subset is not None:
    setname="<{}>".format(options.subset)
    print("select",setname)
    train_set.drop(lambda x: x["ori_words"][0]!=setname,inplace=True)
    test_set.drop(lambda x: x["ori_words"][0]!=setname,inplace=True)
    dev_set.drop(lambda x: x["ori_words"][0]!=setname,inplace=True)
    if options.instances is not None:
        train_set=train_set[:int(options.instances)]
        
# build model and optimizer    
i2t=None
if options.crf:
    #i2t=utils.to_id_list(tag_vocab.word2idx)   
    i2t={}
    for x,y in tag_vocab.word2idx.items():
        i2t[y]=x
    print("use crf:",i2t)

freeze=True if options.freeze else False
model = models.make_CWS(d_model=options.d_model, N=options.N, h=options.h, d_ff=options.d_ff,dropout=options.dropout,word_embedding=init_embedding,bigram_embedding=bigram_embedding,tag_size=len(tag_vocab),task_size=len(task_vocab), word_size=len(uni_vocab), bi_size=len(bi_vocab),crf=i2t,freeze=freeze)

if True:  
    print("multi:",devices)
    model=nn.DataParallel(model,device_ids=devices)    

model=model.to(device)

if options.only_task and options.old_model is not None:
    print("fix para except task embedding")
    for name,para in model.named_parameters():
        if name.find("task_embed")==-1:
            para.requires_grad=False
        else:
            para.requires_grad=True
            print(name)
    
optimizer = optm.NoamOpt(options.d_model, options.factor, 4000,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

optimizer._step=options.step

best_model_file_name = "{}/model.bin".format(root_dir)

train_sampler = BucketSampler(batch_size=options.batch_size, seq_len_field_name='seq_len')
dev_sampler = SequentialSampler()

i2t=utils.to_id_list(tag_vocab.word2idx)   
i2task=utils.to_id_list(task_vocab.word2idx)   
dev_set.set_input("ori_words")
test_set.set_input("ori_words")

word_dic = pickle.load(open(options.dict,"rb"))
    
def tester(model,test_batch,write_out=True):
    res=[]
    prf = utils.CWSEvaluator(i2t)
    prf_dataset = {}
    oov_dataset = {}

    model.eval()
    for batch_x, batch_y in test_batch:
        with torch.no_grad():
            if bigram_embedding is not None:
                out=model(batch_x["task"],batch_x["uni"],batch_x["seq_len"],batch_x["bi1"],batch_x["bi2"])
            else: out = model(batch_x["task"],batch_x["uni"],batch_x["seq_len"])
        out=out["pred"]
        #print(out)
        num=out.size(0)
        out=out.detach().cpu().numpy()
        for i in range(num):
            length=int(batch_x["seq_len"][i])
            
            out_tags=out[i,1:length].tolist()
            sentence = batch_x["ori_words"][i]
            gold_tags = batch_y["tags"][i][1:length].numpy().tolist()
            dataset_name = sentence[0]
            sentence=sentence[1:]
            #print(out_tags,gold_tags)
            assert utils.is_dataset_tag(dataset_name)
            assert len(gold_tags)==len(out_tags) and len(gold_tags)==len(sentence)

            if dataset_name not in prf_dataset:
                prf_dataset[dataset_name] = utils.CWSEvaluator(i2t)
                oov_dataset[dataset_name] = utils.CWS_OOV(word_dic[dataset_name[1:-1]])
                    
            prf_dataset[dataset_name].add_instance(gold_tags, out_tags)
            prf.add_instance(gold_tags, out_tags)
            
            if write_out==True:
                gold_strings = utils.to_tag_strings(i2t, gold_tags)
                obs_strings = utils.to_tag_strings(i2t, out_tags)
                          
                word_list = utils.bmes_to_words(sentence, obs_strings)
                oov_dataset[dataset_name].update(utils.bmes_to_words(sentence, gold_strings), word_list)
                
                raw_string=' '.join(word_list)
                res.append(dataset_name+" "+raw_string+" "+dataset_name)
    
    Ap=0.0
    Ar=0.0
    Af=0.0
    Aoov=0.0
    tot=0
    nw=0.0
    for dataset_name, performance in sorted(prf_dataset.items()):
        p = performance.result()
        if write_out==True:
            nw=oov_dataset[dataset_name].oov()
            logger.info('{}\t{:04.2f}\t{:04.2f}\t{:04.2f}\t{:04.2f}'.format(dataset_name, p[0], p[1], p[2],nw))
        else: logger.info('{}\t{:04.2f}\t{:04.2f}\t{:04.2f}'.format(dataset_name, p[0], p[1], p[2]))
        Ap+=p[0]
        Ar+=p[1]
        Af+=p[2]
        Aoov+=nw
        tot+=1
        
    prf = prf.result()
    logger.info('{}\t{:04.2f}\t{:04.2f}\t{:04.2f}'.format('TOT', prf[0], prf[1], prf[2]))
    if write_out==False:
        logger.info('{}\t{:04.2f}\t{:04.2f}\t{:04.2f}'.format('AVG', Ap/tot, Ar/tot, Af/tot))
    else: logger.info('{}\t{:04.2f}\t{:04.2f}\t{:04.2f}\t{:04.2f}'.format('AVG', Ap/tot, Ar/tot, Af/tot,Aoov/tot))
    return prf[-1], res
           
# start training        
if not options.test:
    if options.old_model:
        # incremental training
        print("Incremental training from old model: {}".format(options.old_model))
        model.load_state_dict(torch.load(options.old_model,map_location="cuda:0"))
              
    logger.info("Number training instances: {}".format(len(train_set)))
    logger.info("Number dev instances: {}".format(len(dev_set)))
    
    train_batch=DataSetIter(batch_size=options.batch_size, dataset=train_set, sampler=train_sampler)
    dev_batch=DataSetIter(batch_size=options.batch_size, dataset=dev_set, sampler=dev_sampler)
    
    best_f1 = 0.
    #bar = utils.Progbar(target=int(options.num_epochs))
    for epoch in range(int(options.num_epochs)):
        logger.info("Epoch {} out of {}".format(epoch + 1, options.num_epochs))
        if epoch == options.flex:
            logger.info("open pretrained embeddings")
            model.module.src_embed.uni_embed.weight.requires_grad = True
            if options.bigram_embeddings is not None:
                model.module.src_embed.bi_embed.weight.requires_grad = True
            optimizer.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=optimizer.rate(), betas=(0.9, 0.98), eps=1e-9)
        train_loss = 0.0
        model.train()   
        tot=0
        t1=time.time()
        #bar = utils.Progbar(target=len(train_set)//options.batch_size+1)
        for batch_x, batch_y in train_batch:
            model.zero_grad()
            if bigram_embedding is not None:
                out=model(batch_x["task"],batch_x["uni"],batch_x["seq_len"],batch_x["bi1"],batch_x["bi2"],batch_y["tags"])
            else: out = model(batch_x["task"],batch_x["uni"],batch_x["seq_len"],tags = batch_y["tags"])
            loss = torch.mean(out["loss"])
            train_loss += loss.item() 
            tot+=1
            loss.backward()
            optimizer.step()
            #bar.update(tot, exact=[("train loss", train_loss)])

        t2=time.time()    
        train_loss = train_loss / tot
        #bar.update(epoch, exact=[("train loss", train_loss)]) 
        logger.info("time: {} loss: {} step: {}".format(t2-t1,train_loss,optimizer._step))
        # Evaluate dev data
        if options.skip_dev:
            logger.info("Saving model to {}".format(best_model_file_name))
            torch.save(model.state_dict(),best_model_file_name)
            continue
   
        model.eval()
        f1, _ =tester(model,dev_batch)
        if f1 > best_f1:
            best_f1 = f1
            logger.info("- new best score!")
            # Serialize model
            if not options.no_model:
                logger.info("Saving model to {}".format(best_model_file_name))
                torch.save(model.state_dict(),best_model_file_name)
                
        elif options.always_model:
            logger.info("Saving model to {}".format(best_model_file_name))
            torch.save(model.state_dict(),best_model_file_name)
                
# Evaluate test data (once)
logger.info("\n")
logger.info("Number test instances: {}".format(len(test_set)))

if not options.skip_dev:
    if options.test:
        model.load_state_dict(torch.load(options.old_model,map_location="cuda:" + options.devi))
    else:
        model.load_state_dict(torch.load(best_model_file_name,map_location="cuda:" + options.devi))
        
for name,para in model.named_parameters():
    if name.find("task_embed")!=-1:
        tm=para.detach().cpu().numpy()
        print(tm.shape)
        np.save("{}/task.npy".format(root_dir),tm)                    
        break
        
test_batch=DataSetIter(test_set,options.batch_size)
_, res=tester(model,test_batch, True)
exit()

with open("{}/testout.txt".format(root_dir), 'w',encoding="utf-16") as raw_writer:
    for sent in res:
        raw_writer.write(sent)
        raw_writer.write('\n')
        
