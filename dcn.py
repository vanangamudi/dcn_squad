import os
import json
import time
import random
from pprint import pprint, pformat

from logger import CMDFilter

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from config import Config

from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
import torch

from trainer import Trainer, Feeder, Predictor
from datafeed import DataFeed, MultiplexedDataFeed
from utilz import tqdm, ListTable

from functools import partial

from collections import namedtuple, defaultdict
import itertools

from utilz import logger
from utilz import PAD, pad_seq, word_tokenize
from utilz import VOCAB
from utilz import Sample
from utilz import LongVar, Var
from vocab import Vocab

import numpy as np
from tokenstring import TokenString

SELF_NAME = os.path.basename(__file__).replace('.py', '')

def load_squad_data(data_path, max_para_len=600, max_ans_len=10):
    dataset = json.load(open(data_path,'r'))
    samples = []
    qn, an = 0, 0
    skipped = 0

    vocabulary = defaultdict(int)
    
    def __(s):
        import unicodedata
        s =  ''.join(
            c for c in unicodedata.normalize('NFKD', s)
            if unicodedata.category(c) != 'Mn'
        )
        return s.replace("``", '"').replace("''", '"')

    try:
        for aid, article in enumerate(tqdm(dataset['data'])):
            for pid, paragraph in enumerate(article['paragraphs']):

                context = TokenString(__(paragraph['context']), word_tokenize).delete_whitespace()
                questions = paragraph['qas']

                for token in context:
                    vocabulary[token] += 1
                    
                
                for qid, qa in enumerate(questions):
                    log.debug('processing: {}.{}.{}'.format(aid, pid, qid))
                    _id = qa['id']
                    q = TokenString(__(qa['question']), word_tokenize).delete_whitespace()
                    a = TokenString(__(qa['answers'][0]['text']), word_tokenize).delete_whitespace()  #simply ignore other answers
                    
                    for token in q:
                        vocabulary[token] += 1

                    indices = context.index(a)
                    if not indices:
                        log.debug(pformat(paragraph['context']))
                        log.debug(pformat(paragraph['qas'][qid]))
                        log.error('{}.{}.{} - "{}" not found in \n"{}"'
                                  .format(aid, pid, qid, a.tokenized_string,
                                          context.tokenized_string))
                        skipped += 1
                        continue
                        
                    a_start, a_end = indices
                    samples.append(Sample(_id, aid, pid, qid, _id, context, q, a, a_start, a_end))
    except:
        skipped += 1
        log.exception('{}'.format(aid))

    print('skipped {} samples'.format(skipped))
    return samples, vocabulary


# ## Loss and accuracy function
def loss(output, target, loss_function=nn.NLLLoss(), scale=1, *args, **kwargs):
    target = Variable(torch.LongTensor(target[0]), requires_grad=False)
    if Config().cuda: target = target.cuda()
    log.debug('i, o sizes: {} {}'.format(output.size(), target.size()))

    return loss_function(output, target)

def accuracy(output, target, *args, **kwargs):
    output = output.max(-1)[1]
    target = Variable(torch.LongTensor(target[0]))
    if Config().cuda: target = target.cuda()
    log.debug('i, o sizes: {} {}'.format(output.size(), target.size()))
    return (output == target).sum().float()/output.size(0)

def repr_function(output, feed, batch_index, VOCAB, LABELS, raw_samples):
    results = []
    output = output.data.max(dim=-1)[1].cpu().numpy()
    indices, (seq, ), (labels,) = feed.nth_batch(batch_index)
    for idx, op, se, la in zip(indices, output, seq, labels):
        #sample = list(filter(lambda x: x.id == int(str(idx).split('.')[0]), raw_samples))[0]
        results.append([ ' '.join(feed.data_dict[idx].tokens), feed.data_dict[idx].sentiment, LABELS[op] ])

    return results

def batchop(datapoints, WORD2INDEX, *args, **kwargs):
    indices = [d.id for d in datapoints]
    context = []
    question = []
    answer_start = []
    answer_end = []
    for d in datapoints:
        context.append([WORD2INDEX[w] for w in d.context])
        question.append([WORD2INDEX[w] for w in d.q])
        answer_start.append([d.a_start])
        answer_end.append([d.a_end])
        
    context = pad_seq(context)
    question = pad_seq(question)

    batch = indices, (np.array(context), np.array(question)), (np.array(answer_start),np.array(answer_end))
    print(batch[1][1].shape)
    return batch

class Base(nn.Module):
    def __init__(self):
        super(Base, self).__init__()
               
        self.log = logging.getLogger('{}.{}'.format(__name__,
                                                    self.__class__.__name__))
        self.size_log = logging.getLogger('{}.{}.{}'.format(__name__,
                                                            self.__class__.__name__,
                                                            'size'))
        self.log.setLevel(logging.DEBUG)
        self.size_log.setLevel(logging.DEBUG)
       
    def cpu(self):
        super(Base, self).cpu()
        return self
    
    def cuda(self):
        super(Base, self).cuda()
        return self
    
    def __(self, tensor, name=''):
        if isinstance(tensor, list) or isinstance(tensor, tuple):
            for i in range(len(tensor)):
                self.__(tensor[i], '{}[{}]'.format(name, i))

        else:
            self.size_log.debug('{} -> {}'.format(name, tensor.size()))
                
        return tensor
    
class Encoder(Base):
    def __init__(self, Config, input_vocab_size):
        super(Encoder, self).__init__()

        self.embed_dim = Config.embed_dim
        self.hidden_dim = Config.hidden_dim
        self.embed = nn.Embedding(input_vocab_size, self.embed_dim)

        self.encode = nn.LSTM(      self.embed_dim, self.hidden_dim, bidirectional=True)
        self.attend = nn.LSTM(4 * self.hidden_dim, self.hidden_dim, bidirectional=True)

        self.linear = nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim)
        
        self.dropout = nn.Dropout(0.1)

        if Config.cuda:
            self.cuda()

    def init_hidden(self, batch_size, cell):
        layers = cell.num_layers
        if cell.bidirectional:
            layers = layers * 2 
        hidden  = Variable(torch.zeros(layers, batch_size, cell.hidden_size))
        context = Variable(torch.zeros(layers, batch_size, cell.hidden_size))
        if Config.cuda:
            hidden  = hidden.cuda()
            context = context.cuda()
        return hidden, context

    def sentinel(self, batch_size=1):
        s = Variable(torch.zeros(batch_size, 1, 2 * self.hidden_dim))
        if Config.cuda:
            s = s.cuda()

        return s
    
    def forward(self, context, question):

        context = LongVar(Config, context)
        question = LongVar(Config, question)
        
        batch_size, context_size  = context.size()
        _         , question_size = question.size()
        
        context  = self.__( self.embed(context),  'context_emb')
        question = self.__( self.embed(question), 'question_emb')

        context  = context.transpose(1,0)
        C, _  = self.__(  self.encode(context, self.init_hidden(batch_size, self.encode)), 'C')
        C     = self.__(  C.transpose(1,0), 'C')
        s     = self.__(  self.sentinel(batch_size), 's')
        C     = self.__(  torch.cat([C, s], dim=1), 'C')

        
        question  = question.transpose(1,0)
        Q, _ = self.__(  self.encode(question, self.init_hidden(batch_size, self.encode)), 'Q')
        Q    = self.__(  Q.transpose(1,0), 'Q')
        s    = self.__(  self.sentinel(batch_size), 's')
        Q    = self.__(  torch.cat([Q, s], dim=1), 'Q')

        
        squashedQ = self.__(  Q.view(batch_size * (question_size + 1), -1), 'squashedQ'    ) 
        transformedQ = self.__(  F.tanh(self.linear(Q))                    , 'transformedQ' )
        Q = self.__(  Q.view(batch_size, question_size + 1, -1) , 'Q' )

        affinity      = self.__(  torch.bmm(C, Q.transpose(1, 2)), 'affinity'  )
        affinity      = F.softmax(affinity)
        context_attn  = self.__( affinity.transpose(1, 2), 'context_attn')
        question_attn = self.__( affinity                , 'question_attn')

        context_question = self.__( torch.bmm(C.transpose(1, 2), question_attn), 'context_question' )
        context_question = self.__( torch.cat([Q, context_question.transpose(1,2)], -1 ), 'context_question' )

        attn_cq       = self.__(  torch.bmm(context_question.transpose(1, 2), context_attn) , 'attn_cq' )
        attn_cq       = self.__(  attn_cq.transpose(1, 2).transpose(0,1), 'attn_cq')
        hidden        = self.__(  self.init_hidden(batch_size, self.attend), 'hidden')
        final_repr, _ = self.__(  self.attend(attn_cq, hidden), 'final_repr')

        return final_repr[:, :-1]  #exclude sentinel
        
        
class Maxout(Base):
    # https://github.com/pytorch/pytorch/issues/805
    def __init__(self, Config, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m
    
    
class HMN(Base):
    def __init__(self, Config):
        super(HMN, self).__init__()
        self.hidden_dim = Config.hidden_dim
        self.pooling_size = Config.pooling_size
        
        self.W_D = nn.Linear(5 * self.hidden_dim, self.hidden_dim)
        
        self.W_1 = Maxout(3 * self.hidden_dim, self.hidden_dim, self.pooling_size)
        self.W_2 = Maxout(    self.hidden_dim, self.hidden_dim, self.pooling_size)
        self.W_3 = Maxout(2 * self.hidden_dim, 1,                self.pooling_size)
        
    def forward(self,u_t,u_s,u_e,h):

        r   = self.__(  F.tanh(self.W_D(torch.cat([u_s,u_e,h],1))), 'r'  )
        m_1 = self.__(  self.W_1(torch.cat([u_t,r],1)), 'm_1'  )
        m_2 = self.__(  self.W_2(m_1), 'm_2'  )
        m_3 = self.__(  self.W_3(torch.cat([m_1,m_2],1)), 'm_3'  )
        
        return m_3
    


def experiment(VOCAB, raw_samples, datapoints=[[], []], eons=1000, epochs=10, checkpoint=1):
    try:
        try:
            model =  Encoder(Config(), len(VOCAB))
            if Config().cuda:  model = model.cuda()
            model.load_state_dict(torch.load('{}.{}'.format(SELF_NAME, 'pth')))
            log.info('loaded the old image for the model')
        except:
            log.exception('failed to load the model')
            model =  Encoder(Config(), len(VOCAB))
            if Config().cuda:  model = model.cuda()
            
        print('**** the model', model)

        name = os.path.basename(__file__).replace('.py', '')
        ids = [Sample._fields.index('squad_id')]
        
        _batchop = partial(batchop, WORD2INDEX=VOCAB)
        train_feed     = DataFeed(name, datapoints[0], ids = ids, batchop=_batchop, batch_size=256)
        test_feed      = DataFeed(name, datapoints[1], ids = ids, batchop=_batchop, batch_size=256)
        predictor_feed = DataFeed(name, datapoints[1], ids = ids, batchop=_batchop, batch_size=128)

        loss_weight=Variable(torch.Tensor([0.1, 1, 1]))
        if Config.cuda: loss_weight = loss_weight.cuda()
        _loss = partial(loss, loss_function=nn.NLLLoss())
        trainer = Trainer(name=name,
                          model=model, 
                          loss_function=_loss, accuracy_function=accuracy, 
                          checkpoint=checkpoint, epochs=epochs,
                          feeder = Feeder(train_feed, test_feed))

        _repr_function=partial(repr_function, VOCAB=VOCAB, raw_samples=raw_samples)
        predictor = Predictor(model=model, feed=predictor_feed, repr_function=_repr_function)
        
        for e in range(eons):
            dump = open('results/experiment_attn.csv', 'a')
            dump.write('#========================after eon: {}\n'.format(e))
            dump.close()
            log.info('on {}th eon'.format(e))

            with open('results/experiment_attn.csv', 'a') as dump:
                results = ListTable()
                for ri in range(predictor_feed.num_batch):
                    output, _results = predictor.predict(ri)
                    results.extend(_results)
                dump.write(repr(results))
            if not trainer.train():
                raise Exception
        
    except :
        log.exception('####################')
        trainer.save_best_model()

        return locals()



    
import sys
import pickle
if __name__ == '__main__':

    if sys.argv[1]:
        log.addFilter(CMDFilter(sys.argv[1]))

    flush = False
    if flush:
        log.info('flushing...')
        dataset, vocabulary = load_squad_data('dataset/train-v1.1.json')
        pickle.dump([dataset, dict(vocabulary)], open('train.squad', 'wb'))
    else:
        dataset, _vocabulary = pickle.load(open('train.squad', 'rb'))
        vocabulary = defaultdict(int)
        vocabulary.update(_vocabulary)
        
    log.info('dataset size: {}'.format(len(dataset)))
    #og.info('dataset[:10]: {}'.format(pformat(dataset[:10])))
    log.info('vocabulary: {}'.format(len(vocabulary)))
    encoder = Encoder(Config, 100)
    
    VOCAB = Vocab(vocabulary, VOCAB)
    
    if 'train' in sys.argv:
        labelled_samples = dataset
        pivot = int( Config().split_ratio * len(labelled_samples) )
        random.shuffle(labelled_samples)
        train_set, test_set = labelled_samples[:pivot], labelled_samples[pivot:]
        
        train_set = sorted(train_set, key=lambda x: len(x.context))
        test_set = sorted(test_set, key=lambda x: len(x.context))
        
        experiment(VOCAB, dataset, datapoints=[train_set, test_set])
        
    if 'predict' in sys.argv:
        model =  BiLSTMDecoderModel(Config(), len(VOCAB),  len(LABELS))
        if Config().cuda:  model = model.cuda()
        model.load_state_dict(torch.load('{}.{}'.format(SELF_NAME, '.pth')))
        start_time = time.time()
        strings = sys.argv[2]
        
        s = [WORD2INDEX[i] for i in word_tokenize(strings)] + [WORD2INDEX['PAD']]
        e1, e2 = [WORD2INDEX['ENTITY1']], [WORD2INDEX['ENTITY2']]
        output = model(s, e1, e2)
        output = output.data.max(dim=-1)[1].cpu().numpy()
        label = LABELS[output[0]]
        print(label)

        duration = time.time() - start_time
        print(duration)
