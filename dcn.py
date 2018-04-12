import os
import json
import time
import random
from pprint import pprint, pformat

from logger import CMDFilter

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(name)s.%(funcName)s >>   %(message)s")
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
from utilz import LongVar, Var, init_hidden
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

def repr_function(output, feed, batch_index, VOCAB, raw_samples):
    results = []
    output = output.data.max(dim=-1)[1].cpu().numpy()
    indices, (context, question), (a_start, a_end) = feed.nth_batch(batch_index)
    for idx, op, a_s, a_e in zip(indices, output, a_start, a_end):
        sample = feed.data_dict[idx]
        results.append([ ' '.join(sample.context),
                         ' '.join(sample.q),
                         ' '.join(sample.context[sample.a_start: sample.a_end]),
                         ' '.join(sample.context[          op[0]: op[1]       ])
        ])

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
    def __init__(self, Config, name):
        super(Base, self).__init__()
        self._name = name
        self.log = logging.getLogger(self._name)
        size_log_name = '{}.{}'.format(self._name, 'size')
        self.log.info('constructing logger: {}'.format(size_log_name))
        self.size_log = logging.getLogger(size_log_name)
        self.size_log.info('size_log')
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

    def name(self, n):
        return '{}.{}'.format(self._name, n)
    
class Encoder(Base):
    def __init__(self, Config, name, input_vocab_size):
        super(Encoder, self).__init__(Config, name)

        self.embed_size = Config.embed_size
        self.hidden_size = Config.hidden_size
        self.embed = nn.Embedding(input_vocab_size, self.embed_size)

        self.encode = nn.LSTM(     self.embed_size, self.hidden_size, bidirectional=True)
        self.attend = nn.LSTM(4 * self.hidden_size, self.hidden_size, bidirectional=True)

        self.linear = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size)
        
        self.dropout = nn.Dropout(0.1)

        if Config.cuda:
            self.cuda()

    def sentinel(self, batch_size=1):
        s = Variable(torch.zeros(batch_size, 1, 2 * self.hidden_size))
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
        C, _  = self.__(  self.encode(context, init_hidden(batch_size, self.encode)), 'C')
        C     = self.__(  C.transpose(1,0), 'C')
        s     = self.__(  self.sentinel(batch_size), 's')
        C     = self.__(  torch.cat([C, s], dim=1), 'C')

        
        question  = question.transpose(1,0)
        Q, _ = self.__(  self.encode(question, init_hidden(batch_size, self.encode)), 'Q')
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
        hidden        = self.__(  init_hidden(batch_size, self.attend), 'hidden')
        final_repr, _ = self.__(  self.attend(attn_cq, hidden), 'final_repr')

        return final_repr[:, :-1]  #exclude sentinel
        
        
class Maxout(Base):
    # https://github.com/pytorch/pytorch/issues/805
    def __init__(self, Config, name, input_size, output_size, pool_size):
        super(Maxout, self).__init__(Config, name)
        self.input_size, self.output_size, self.pool_size = input_size, output_size, pool_size
        self.lin = nn.Linear(input_size, output_size * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.output_size
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m = self.__( out.view(*shape), 'm' )
        m, _ = m.max(max_dim)
        return m
    
    
class HMN(Base):
    def __init__(self, Config, name):
        super(HMN, self).__init__(Config, name)
        self.hidden_size = Config.hidden_size
        self.pooling_size = Config.pooling_size
        
        self.linear = nn.Linear(5 * self.hidden_size, self.hidden_size)
        
        self.W_1 = Maxout(Config, self.name('W_1'), 3 * self.hidden_size, self.hidden_size, self.pooling_size)
        self.W_2 = Maxout(Config, self.name('W_2'),    self.hidden_size, self.hidden_size, self.pooling_size)
        self.W_3 = Maxout(Config, self.name('W_3'), 2 * self.hidden_size, 1,               self.pooling_size)
        
    def forward(self, ut, start_repr, end_repr, hidden):

        self.__(start_repr, 'start_repr')
        self.__(end_repr, 'end_repr')
        self.__(hidden, 'hidden')
        
        r = self.__(  torch.cat( [start_repr, end_repr, hidden], dim=1 ), 'r'  )
        r = self.__(  F.tanh(self.linear(r))                        , 'r'  )

        ut_r = self.__(  torch.cat([ut, r], dim=1), 'ut_r')
        
        m1 = self.__(  self.W_1(ut_r), 'm1'  )
        m2 = self.__(  self.W_2(m1), 'm2'  )

        m3 = self.__(  torch.cat([m1, m2], dim=1), 'm3')
        return  self.__(  self.W_3(m3), 'm3'  )
        

class PtrDecoder(Base):
    def __init__(self, Config, name):
        super(PtrDecoder, self).__init__(Config, name)
        
        self.hidden_size  = Config.hidden_size
        self.pooling_size = Config.pooling_size
        self.max_iter     = Config.max_iter

        self.decode = nn.LSTMCell(4 * self.hidden_size, self.hidden_size)
        
        self.find_start = HMN(Config, self.name('find_start'))
        self.find_end   = HMN(Config, self.name('find_end'))

        self.dropout = nn.Dropout(Config.dropout)

        if Config.cuda:
            self.cuda()
        
    def forward(self, encoded_repr):
        batch_size, seq_len, hidden_size = encoded_repr.size()
        start_repr = encoded_repr[:, 0, :]
        end_repr   = encoded_repr[:, 1, :]

        hidden = self.__(  init_hidden(batch_size, self.decode), 'hidden'  )
        for i in range(self.max_iter):
            starts, ends = [], []

            for ut in encoded_repr.transpose(0, 1):
                start = self.__( self.find_start(ut, start_repr, end_repr, hidden[0]), 'start')
                starts.append(start)

            starts = self.__( torch.stack(starts), 'starts')
            start = starts.data.max(1)[1]
            start_repr = torch.cat([  encoded_repr[i][ start[i] ] for i in range(seq_len)   ])
                
            for ut in encoded_repr.transpose(0, 1):
                end = self.__( self.find_end(ut, end_repr, end_repr, hidden[0]), 'end')
                ends.append(end)

            ends = self.__( torch.stack(ends), 'ends')
            end = ends.data.max(1)[1]
            end_repr = torch.cat([  encoded_repr[i][end[i]] for i in range(seq_len)   ])

            input_ = self.__( torch.cat([start_repr, end_repr], -1), 'input_')
            hidden = self.__( self.decode(input_, hidden), 'hidden')

        return starts, ends


class DCN(Base):
    def __init__(self, Config, name, input_vocab_size):
        super(DCN, self).__init__(Config, name)
        self.encode = Encoder(Config, self.name('encode'), input_vocab_size)
        self.decode = PtrDecoder(Config, self.name('decode'))

    def forward(self, context, question):
        return self.decode( self.encode(context, question) )
        
            
def experiment(VOCAB, raw_samples, datapoints=[[], []], eons=1000, epochs=10, checkpoint=1):
    try:
        try:
            model =  DCN(Config(), 'model', len(VOCAB))
            if Config().cuda:  model = model.cuda()
            model.load_state_dict(torch.load('{}.{}'.format(SELF_NAME, 'pth')))
            log.info('loaded the old image for the model')
        except:
            log.exception('failed to load the model')
            model =  DCN(Config(), 'model', len(VOCAB))
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
    #log.info('dataset[:10]: {}'.format(pformat(dataset[:10])))
    log.info('vocabulary: {}'.format(len(vocabulary)))
    
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
