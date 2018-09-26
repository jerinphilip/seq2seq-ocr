#!/usr/bin/env python
# coding: utf-8


import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import funcy as fy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import string
import sys
from pympler import muppy, summary


from s2s_ocr.synthesis import FontImageRender
from s2s_ocr.data import RandomCharsDataset        
from s2s_ocr.models import Encoder, AttnDecoder, Seq2Seq
from s2s_ocr.modules import IdentityEmbedding
from s2s_ocr.criterions import TNLLLoss


dataset = RandomCharsDataset(string.digits + string.ascii_letters + string.punctuation + ' ', size=8192, fonts=["Serif 22"])
loader = DataLoader(dataset, batch_size=512, collate_fn=RandomCharsDataset.collate, pin_memory=True)
device = torch.device("cuda")


x, z, lx, lz = next(iter(loader))
plt.imshow(x[:, 0, :].transpose(0, 1))
plt.title(dataset.decode(z[:, 0]))
plt.axis('off')


id_embed = IdentityEmbedding(input_size=dataset.images_height)
d_embed = nn.Embedding(dataset.classes, 64)    
encoder = Encoder(id_embed, hidden_size=64)
decoder = AttnDecoder(d_embed, hidden_size=64,output_size=dataset.classes)
seq2seq = Seq2Seq(encoder, decoder, dataset.vocab)
seq2seq.to(device)


opt = optim.Adam(seq2seq.parameters())
criterion = TNLLLoss()

seq2seq.load("checkpoints/best.pt")
MAX_LENGTH = 60
interval = 5
max_epochs = 0
for epoch in range(max_epochs):
    total_loss = 0
    batches = 0
    for x, z, lx, lz in loader:
        x = x.to(device)
        z = z.to(device)
        opt.zero_grad()
        loss = 0
        outputs = seq2seq(x, lx, z, lz)
        batches += 1     
        loss = criterion(outputs, z[1:])
        loss.backward()
        total_loss += loss.item()
        opt.step()

    if epoch%interval == 0 or epoch + 1  == max_epochs:
        print("Epoch", epoch, "Loss:", total_loss/(batches))
        #checkpoint(encoder, decoder, "model.chpt".format(epoch))
        seq2seq.save("checkpoints/best.pt")

from s2s_ocr.report_hooks import TransBatchCompileHook
reporter = TransBatchCompileHook()



test = Seq2Seq(encoder, decoder, dataset.vocab, report_hook=reporter)
with torch.no_grad():
    x, z, lx, lz = next(iter(loader))
    x = x.to(device)
    z = z.to(device)
    outputs = test(x, lx, None, None)



def visualize_attention(inputs, truths, preds, attn):    
    print(dataset.decode(preds))
    plt.figure(figsize=(64, 8))
    plt.subplot(211)
    H, W = inputs.size()
    plt.imshow(inputs.transpose(0, 1))
    plt.subplot(212)
    B, H = attn.size()
    cax = plt.matshow(attn, cmap='bone', fignum=False, aspect='equal', extent=(0, H, W, 0))
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    plt.tight_layout()
    return plt

for i in range(len(reporter)):
    inputs, truths, preds, attns = reporter[i]
    plt = visualize_attention(inputs, truths, preds, attns)
    name = dataset.decode(preds).replace("<pad>", "").replace("</s>", "")
    plt.savefig('outputs/{}.png'.format(name))
