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
             
V = [[0]*len(new_inds) for i in seq]
V[0][0] = 1
Ptr = [[0]*len(new_inds) for i in seq]
#build V an ptr

tr_mat_np = np.array(tr_mat)
with np.errstate(divide='ignore'):
    tr_mat_np_log = np.log(tr_mat)
    Vl = np.log(V)
    Ptrl = np.log(Ptr)
for i, nuq in enumerate(seq[1:]):
    #not log space (for self check):
    maxim_pr = np.amax(np.multiply(tr_mat_np.T,(V[i])),axis = 1)
    V[i+1] = np.multiply(maxim_pr,np.array(em_mat1[nuq]))
    Ptr[i+1] = np.argmax(np.multiply(tr_mat_np.T,(V[i])), axis = 1)
    # log space (for self check):
    maxim_pr_l = np.amax(tr_mat_np_log.T+Vl[i],axis = 1)
    Ptrl[i+1] = np.argmax(tr_mat_np_log.T+Vl[i],axis = 1)
    Vl[i+1] = maxim_pr+np.array(em_mat1[nuq])
pn = np.argmax(Vl[-1])
pn1 = Ptrl[-1][pn]
motif = new_inds[int(pn1)][0]+new_inds[int(pn)][0]
for i in range(len(seq[1:-3])):
    i = len(seq) - i - 3   
    next_ind = Ptrl[i][int(pn1)]
    pn1= next_ind
    motif = new_inds[int(next_ind)][0] + str(motif)
# V = np.array(V).T
# Ptr = np.array(Ptr).T
# VnMax = np.argmax(V[:, V.shape[1] -1])
# motif = ''
# col = Ptr.shape[1] -1
# row = VnMax
# while col != 0:
#      motif += new_inds[post_mat[i].index(max(post_mat[i]))]
#     if (row < 2) | (row >= (len(Ptr) - 2)):
#         motif += 'B'
#     else:
#         motif += 'M'
#     row = Ptr[row,col]
#     col -= 1
    
# motif = motif[::-1]
# motif = motif[:-1]
seq = seq[1:-1]
for i in range(len(seq)//50 + 1):
    print(motif[i*50:(i+1)*50],seq[i*50:(i+1)*50], sep = '\n', end='\n\n')


