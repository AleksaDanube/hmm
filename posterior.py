# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 21:35:59 2020

@author: adanov
"""
import argparse
from Bio import SeqIO
import csv
import pandas as pd
import os
import numpy as np
from scipy.special import logsumexp
forward_mat = forward()
back_mat = backward()
post_mat = (np.array(forward_mat)+np.array(back_mat))-logsumexp(forward_mat[-2])
motif = ''
for i, nuq in enumerate(seq):
    motif += new_inds[post_mat[i].index(max(post_mat[i]))]
    
for i in range(len(seq)//50 + 1):
    print(motif[i*50:(i+1)*50],seq[i*50:(i+1)*50], sep = '\n', end='\n\n')
