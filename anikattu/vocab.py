
import logging
from pprint import pprint, pformat
logging.basicConfig(format="%(levelname)-8s:%(filename)s.%(funcName)20s >>   %(message)s")
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class Vocab:

    def __init__(self, vocab, special_tokens, max_size=None, sort_key=None):

        log.info('Constructiong vocabuluary object...')
        self.vocab = vocab

        vocab = sorted(vocab.items(), key=lambda x: x[1])
        if max_size: vocab = vocab[:max_size]
        index2word = vocab
        if sort_key:
            index2word = sorted(index2word, key=sort_key)

        self.index2word = special_tokens + index2word
        self.word2index = {w:i for i, w in enumerate(self.index2word)}

        log.info('number of word in index2word and word2index: {} and {}'
                 .format(len(self.index2word), len(self.word2index)))
        
    def __getitem__(self, key):
        if type(key) == int:
            return self.index2word[key]
        elif type(key) == str:
            if key in self.word2index:
                return self.word2index[key]
            else:
                return self.word2index['UNK']


    def __len__(self):
        return len(self.index2word)

    def extend(self, words):        
        self.word2index.update(
            {
                w  :   i + len(self.index2word)
                for i, w in enumerate(words)
            })
        
        self.index2word += words
