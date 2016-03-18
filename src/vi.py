import sys
import numpy as np
import update
from sklearn.preprocessing import normalize

# Input: 
# 	Documents {B_1,...,B_D}
# Output:
# 	Variational parameters {u, u_prime, rho, alpha, eta, z}	
# Variables:
#	W:			Words in each documents, W[d][n] is the n-th word in the d-th doc
#	K:			Total number of topics
#	V:			Size of the vocabulary
#	D:			Total number of documents
#	N:			A list where N[d] = Number of words in document d
# 	word_dim:	        Dimension of word embedding space
# 	doc_dim:	        Dimension of document embedding space
#	word_emb:			Word-embedding results for each word
#	word2idx:			Index of each word in the word_emb vector

K = 10 # number of topics
V = 0 # 
D = 0 #
W = list()
N = list()
word_dim = 100
doc_dim = 100 # embedding space dimension of document
word2idx = dict() # mapping from a word to it's index in the vocabulary
word_emb = list()

def init_vars(D, K, V, N, doc_dim, word_dim):
    # initialize Z s.t. Z_dn is a vector of size K as parameters for a categorical distribution
    # Z is the variational distribution of q(z_dn), q(z_dn = k) = Z(d, n, k)
    Z = list()
    for d in range(D):
        Z.append([normalize(np.random.uniform(0, 1, (K, 1)), 'l1') for i in range(N[d])])

    # initialize Eta with parameters Sigma (D * K) and mu (D * K) that defines a multivariate Gaussian distribution
    # Eta_{dk} ~ Normal(mu[d][k], Sigma[d][k])
    Eta = dict(Sigma = [np.random.rand(K, 1) for d in range(D)], mu = [np.random.rand(K, 1) for d in range(D)])

    # initialize A with parameters Sigma (doc_dim * doc_dim) and mu (D * doc_dim) such that
    # A_d ~ Normal(mu[d], Sigma)
    A = dict(Sigma = np.diag(np.random.rand(doc_dim, 1)), mu = [np.random.rand(doc_dim, 1) for d in range(D)])

    # initialize Rho with parameters Sigma (K * V) and mu (K * V) that defines a multivariate Gaussian distribution
    # Rho_{kw} ~ Normal(mu[k][w], Sigma[k][w])
    Rho = dict(Sigma = [np.random.rand(V, 1) for k in range(K)], mu = [np.random.rand(V, 1) for k in range(K)])

    # initialize U_prime with parameters Sigma (word_dim * word_dim) and mu (K * word_dim) such that
    # U_prime_k ~ Normal(mu[k], Sigma)
    U_prime = dict(Sigma = np.diag(np.random.rand(word_dim, 1)), mu = [np.random.rand(word_dim, 1) for k in range(K)])

    # initialize U with parameters Sigma (doc_dim * doc_dim) and mu (K * doc_dim) such that
    # U_k ~ Normal(mu[k], Sigma)
    U = dict(Sigma = np.diag(np.random.rand(doc_dim, 1)), mu = [np.random.rand(doc_dim, 1) for k in range(K)])

    # Xi_KW and Alpha_K are the auxiliary variable related to the lower bound used for q(z_dn)
    Xi_KW = [np.random.rand(V, 1) for i in range(K)]
    Alpha_K = np.random.rand(K, 1)

    # Xi_DK and Alpha_D are the auxiliary variable related to the lower bound used for q(eta_d)
    Xi_DK = [np.random.rand(K, 1) for i in range(D)]
    Alpha_D = np.random.rand(D, 1)

    return Z, Eta, A, Rho, U_prime, U, Xi_KW, Alpha_K, Xi_DK, Alpha_D

def load_documents(word_emb_file, corpus_file):
    filein = open(word_emb_file, 'r')
    filein.readline()
    index = 0
    idx2word = dict()
    for line in filein:
        vals = line.strip().split()
        # build the word2idx dictionary
        word2idx[vals[0]] = index
        idx2word[index] = vals[0]
        # store the word-embedding results
        word_emb.append(vals[1: ])
        index += 1
    filein.close()

    # read in the corpus file
    filein = open(corpus_file, 'r')
    for doc in filein:
        words = doc.strip().split()
        W.append(words)
        N.append(len(words))
    filein.close()

    # setting the vocabulary size
    V = len(word2idx)

def run():
    # initialize all variables
    word_emb_file = '???'
    corpus_file = '???'
    l = 1
    c = 1
    kappa = 1
    beta = 1
    gamma = 1
    load_documents(word_emb_file, corpus_file)
    (Z, Eta, A, Rho, U_prime, U, Xi_KW, Alpha_K, Xi_DK, Alpha_D) = init_vars(D, K, V, N, doc_dim, word_dim)

    # TODO precompute Sigma^{(u')*} by Eq. 9
    compute_u_prime_sigma(U_prime, beta, l, word_emb, V)

    while true: # while not converge
        # TODO sample a batch of document B
        # TODO 
        for d in B:
            for n in N[d]:
                update.update_z(d, n, Z, Eta, Rho, Xi_KW, Alpha_K, W, word2idx, K, V)
            # update Eta
            update.update_eta(d, Eta, Xi_DK, Alpha_D, U, A, Z, gamma, N, K)
            # update A
            update.update_a(d, A, U, Eta, c, gamma, K)
            if certain_interval:
                # TODO update auxiliary variables ksi_d and alpha_d by Eq 2 and Eq 3


        # Tianshu Ren starts here TODO
        for k in K:
            # update rho
            update.update_rho(k, Rho, q_Z, beta, word_emb, U_prime, Alpha_K, Xi_KW)
            # TODO update u_k_tild by Eq.10
            # update U_prime
            update.update_u_prime(k, U_prime, beta, word_emb, Rho)
            if certain_interval():
                # TODO update xi_k by Eq. 5
                # TODO update alpha_k by Eq. 6

        if converge:
            break

if __name__ == "__main__":
    run()
