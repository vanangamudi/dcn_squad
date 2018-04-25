from config import Config
from pprint import pprint, pformat

import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

import torch
from torch import nn
from torch.autograd import Variable

from collections import namedtuple, defaultdict
"""
from nltk.tokenize import WordPunctTokenizer
word_punct_tokenizer = WordPunctTokenizer()
word_tokenize = word_punct_tokenizer.tokenize
"""

from anikattu.tokenizer import word_tokenize

VOCAB =  ['PAD', 'UNK', 'EOS']
PAD = VOCAB.index('PAD')

"""
    Local Utilities, Helper Functions

"""
BoundarySample = namedtuple('BoundarySample', ['id', 'aid', 'pid', 'qid', 'squad_id', 'context', 'q', 'a', 'a_start', 'a_end'])
SequenceSample = namedtuple('SequenceSample', ['id', 'aid', 'pid', 'qid', 'squad_id', 'context', 'q', 'a', 'a_positions'])
