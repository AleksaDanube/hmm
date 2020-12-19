# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 23:16:41 2020

@author: adanov
"""
import argparse
from Bio import SeqIO
import csv
import pandas as pd
import os
import numpy as np
from scipy.special import logsumexp
tsv_file = open('emis.tsv')
read_tsv = pd.read_csv(tsv_file, delimiter="\t", header = 0)
em_mat = read_tsv
#forward(args.p, args.q, args.seq, read_tsv)
p = 0.1
q= 0.99
seq = 'CCAAAATT'
#preproc add start , end chars to the seq
seq = ''.join(['^',seq,'$'])
#preprocessing update em_mat
len_of_motif = len(em_mat.index)
em_mat['$'] = [0]*len_of_motif
em_mat['^'] = [0]*len_of_motif

df1 = pd.DataFrame([[0, 0, 0, 0, 0, 1], [0.25, 0.25, 0.25, 0.25, 0, 0]], columns=['A','C','G','T', '$', '^'])
df2 = pd.DataFrame([[0.25, 0.25, 0.25, 0.25, 0, 0],[0, 0, 0, 0, 1, 0]], columns=['A','C','G','T', '$', '^'])
em_mat1 = pd.concat([df1, em_mat])
em_mat1 = em_mat1.append(df2)

motif_inds = ['M'+str(i+1) for i in range(len_of_motif)]
new_inds = ['BS', 'B1']
new_inds.extend(motif_inds)
new_inds.extend(['B2', 'BE'])
em_mat1.index = new_inds

#build transition matrix in sparse way

motif_pr = {(s, 'M'+str(i+2)):1 for i, s in enumerate(motif_inds[:-1])}
tr_mat_sparse = {('BS','B1'):q, ('BS','B2'):(1-q), ('B1','B1'): (1-p), ('B1', 'M1'): p, (motif_inds[-1], 'B2'):1, ('B2','B2'):(1-p), ('B2', 'BE'):p}
tr_mat_sparse.update(motif_pr)

#build transition matrix:
tr_mat = [[0]*len(new_inds) for i in new_inds]
for i, s1 in enumerate(new_inds):
    for j, s2 in enumerate(new_inds):
        if (s1,s2) in tr_mat_sparse:
            tr_mat[i][j] = tr_mat_sparse[(s1,s2)]

#initiate forward matrix
             
forward_mat = [[0]*len(new_inds) for i in seq]

#build forward matrix

forward_mat[0][new_inds.index('BS')] = 1
tr_mat_np = np.array(tr_mat)
with np.errstate(divide='ignore'):
    tr_mat_np_log = np.log(tr_mat)
    forward_mat_log = np.log(forward_mat)
for i, nuq in enumerate(seq[1:]):
    #not log space (for self check):
    forward_mat[i+1] = np.multiply((tr_mat_np.transpose().dot(forward_mat[i])),np.array(em_mat1[nuq]))
    #log space:
    with np.errstate(divide='ignore'):
        step1_log = (tr_mat_np_log.T + np.array(forward_mat_log[i]))
        forward_mat_log[i+1] = [logsumexp(col) for col in step1_log]+np.log(em_mat1[nuq])
      
#calculate the log likelihood for the sequence :

llog_seq = logsumexp(forward_mat_log[-1])
print(np.log(sum(forward_mat[i+1])), llog_seq)      
f_mat_log = np.array(forward_mat_log).T


