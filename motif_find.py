import argparse
from Bio import SeqIO
import csv
import pandas as pd
import os
import numpy as np
from scipy.special import logsumexp

def forward(p, q, seq, em_mat):
    
    
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
    #print(np.log(sum(forward_mat[i+1])), llog_seq)      
    #f_mat_log = np.array(forward_mat_log).T
    return(forward_mat_log, llog_seq)
        
def backward(p, q, seq, em_mat):
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
                 
    back_mat = [[0]*len(new_inds) for i in seq]
    
    #build forward matrix
    
    back_mat[-1][len(back_mat[-1])-1] = 1
    tr_mat_np = np.array(tr_mat)
    with np.errstate(divide='ignore'):
        tr_mat_np_log = np.log(tr_mat)
        back_mat_log = np.log(back_mat)
    for i, nuq in enumerate(seq[-2::-1]):
        #not log space (for self check):
        i = len(seq) - i-1
        
        back_mat[i-1] = tr_mat_np.dot(np.multiply(np.array(em_mat1[seq[i]]),np.array(back_mat[i])))
        #log space:
        with np.errstate(divide='ignore'):
            step1_log = (tr_mat_np_log + np.array(back_mat_log[i]) + np.log(np.array(em_mat1[seq[i]])))
            back_mat_log[i-1] = [logsumexp(col) for col in step1_log]
        
    #calculate the log likelihood for the sequence :
    
    llog_seq = back_mat_log[0][0]


    return back_mat_log, llog_seq
def posterior(seq, em_mat, forward_mat, back_mat, llh):
    len_of_motif = len(em_mat.index)
    post_mat = (np.array(forward_mat)+np.array(back_mat))-llh
    motif_inds = ['M'+str(i+1) for i in range(len_of_motif)]
    new_inds = ['BS', 'B1']
    new_inds.extend(motif_inds)
    new_inds.extend(['B2', 'BE'])
    motif = ''
    with np.errstate(divide='ignore'):
        for i, nuq in enumerate(seq):
           motif += (str(new_inds[list(post_mat[i+1]).index(max(post_mat[i+1]))]))[0]
    
    for i in range(len(seq)//50 + 1):
        print(motif[i*50:(i+1)*50],seq[i*50:(i+1)*50], sep = '\n', end='\n\n')


def viterbi(p, q, seq, em_mat):
    #preprocessing of seq
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
    
    #initiate viterbi matrix
                 
    V = [[0]*len(new_inds) for i in seq]
    V[0][0] = 1
    Ptr = [[0]*len(new_inds) for i in seq]
    
    #build V and ptr
    
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
        # log space:
        maxim_pr_l = np.amax(tr_mat_np_log.T+Vl[i],axis = 1)
        Ptrl[i+1] = np.argmax(tr_mat_np_log.T+Vl[i],axis = 1)
        Vl[i+1] = maxim_pr_l+np.array(em_mat1[nuq])
    pn = np.argmax(Vl[-1])
    pn1 = Ptrl[-1][pn]
    motif = new_inds[int(pn1)][0]+new_inds[int(pn)][0]
    for i in range(len(seq[1:-3])):
        i = len(seq) - i - 3   
        next_ind = Ptrl[i][int(pn1)]
        pn1= next_ind
        motif = new_inds[int(next_ind)][0] + str(motif)
    seq = seq[1:-1]
    for i in range(len(seq)//50 + 1):
        print(motif[i*50:(i+1)*50],seq[i*50:(i+1)*50], sep = '\n', end='\n\n')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', help='Algorithm (e.g. viterbi)', required=True)
    parser.add_argument('seq', help='A sequence over the alphabet [A,C,G,T] (e.g. ACTGGACTACGTCATGCA)')
    parser.add_argument('initial_emission', help='Path to emission table (e.g. initial_emission.tsv)')
    parser.add_argument('p', help='transition probability p (e.g. 0.01)', type=float)
    parser.add_argument('q', help='transition probability q (e.g. 0.5)', type=float)
    args = parser.parse_args()
    tsv_file = open(args.initial_emission)
    read_tsv = pd.read_csv(tsv_file, delimiter="\t", header = 0)
    em_mat = read_tsv
    seq = args.seq

    if args.alg == 'viterbi':
        viterbi(args.p, args.q, seq, em_mat)

    elif args.alg == 'forward':
        forward_table, llh_f = forward(args.p, args.q, seq, em_mat)
        print(llh_f)

    elif args.alg == 'backward':
        backward_table, llh_b = backward(args.p, args.q, seq, em_mat)
        print(llh_b)

    elif args.alg == 'posterior':
        forward_table, llh_f = forward(args.p, args.q, seq, em_mat)
        backward_table, llh_b = backward(args.p, args.q, seq, em_mat)
        posterior(seq, em_mat, forward_table, backward_table, llh_f)

if __name__ == '__main__':
    main()
