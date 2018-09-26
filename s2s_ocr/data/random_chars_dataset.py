from torch.utils.data import Dataset
import random
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from ..synthesis import FontImageRender
from ..utils import Vocab

class RandomCharsDataset(Dataset):
    def __init__(self, characters, size, fonts):
        self.characters = sorted(list(characters))
        self.size = size
        self.vocab = Vocab()
        for character in self.characters:
            self.vocab.add(character)
        
        self.classes = len(self.vocab)
        self.images_height = 64
        self.create_render_engine(fonts)
    
    def create_render_engine(self, fonts):
        self.fonts = fonts
        self.render = {font : FontImageRender(font, width=1000, height=self.images_height) \
                       for font in fonts }
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        random.seed(index)
        length = random.randint(5, 10)
        text = ''.join(random.choice(self.characters) for _ in range(length))
        font = random.choice(self.fonts)
        image = self.render[font](text)
        text = [self.vocab.special.bos] + list(text) + [self.vocab.special.eos]
        tgt = np.array([self.vocab[c] for c in text])
        return (image.transpose(), tgt)
    
    def decode(self, tgt):
        xs = tgt.tolist()
        decoded = ''.join([self.vocab.i2w[i] for i in xs])
        return decoded
    
    @staticmethod
    def collate(data):
        """ List of (img.transpose(), tgt) => padded packed sequence """
                 
        data = sorted(data, key=lambda x: x[0].shape[0], reverse=True)
        srcs, tgts = list(zip(*data))        
        
        src_lengths = [src.shape[0] for src in srcs]
        tgt_lengths = [tgt.shape[0] for tgt in tgts]
        

        src_tensors = [torch.tensor(src).float() for src in srcs]
        src = pad_sequence(src_tensors)
        
        sorted_idxs = np.argsort(-1*np.array(tgt_lengths))
        unsort_idxs = np.argsort(sorted_idxs)
        
        #print(sorted_idxs, unsort_idxs)
        
        tgt_tensors = [torch.tensor(tgts[i]).long() for i in sorted_idxs]
        
        tgt = pad_sequence(tgt_tensors)
        tgt = tgt[:, unsort_idxs]
        return (src, tgt, src_lengths, tgt_lengths)
