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

def init_vars():
    # initialize Z s.t. Z_dn is a vector of size K as parameters for a categorical distribution
    # Z is the variational distribution of q(z_dn), q(z_dn = k) = Z(d, n, k)
    Z = list()
    for d in range(D):
        Z.append([normalize(np.random.uniform(0, 1, (1, K)), 'l1') for i in range(N[d])])

    # initialize Eta s.t. Eta_d contains two fields "mu" (1 * K) and "Sigma" (K*K) that specifies a multivariate Gaussian
    Eta = dict(Sigma = [np.random.rand(1, K) for d in range(D)], mu = [np.random.rand(1, K) for d in range(D)])
    # Eta = list()
    # for d in range(D):
    #     Eta.append(gen_normalparams(K))

    # initialize A s.t. A_d contains two fields "mu" (1 * doc_dim) and "Sigma" (doc_dim * doc_dim) that specifies a multivariate Gaussian
    A = dict(Sigma = np.diag(np.random.rand(1, doc_dim)), mu = [np.random.rand(1, doc_dim) for d in range(D)])
    # A = list()
    # for d in range(D):
    #     A.append(gen_normalparams(doc_dim))


    # initialize Rho s.t. Rho_k contains two fields "mu" (1 * V) and "Sigma" (V * V) that specifies a multivariate Gaussian
    # Rho = dict(Sigma = np.diag(np.random.rand(1, V)), mu = [np.random.rand(1, V) for k in range(K)])
    Rho = dict(Sigma = [np.random.rand(1, V) for k in range(K)], mu = [np.random.rand(1, V) for k in range(K)])
    # Rho = list()
    # for k in range(K):
    #     Rho.append(gen_normalparams(V))

    # initialize U_prime s.t. U_prime_k contains two fields "mu" (1 * word_dim) and
    # "Sigma" (word_dim * word_dim) that specifies a multivariate Gaussian
    U_prime = dict(Sigma = np.diag(np.random.rand(1, word_dim)), mu = [np.random.rand(1, word_dim) for k in range(K)])
    # U_prime = list()
    # for k in range(K):
    #     U_prime.append(gen_normalparams(word_dim))

    # initialize U s.t. U_k contains two fields "mu" (1 * doc_dim) and "Sigma" (doc_dim, doc_dim)
    U = dict(Sigma = np.diag(np.random.rand(1, doc_dim)), mu = [np.random.rand(1, doc_dim) for k in range(K)])
    # U = list()
    # for k in range(K):
    #     U.append(gen_normalparams(doc_dim))

    # Xi_KW and Alpha_K are the auxiliary variable related to the lower bound used for q(z_dn)
    Xi_KW = [np.random.rand(1, V) for i in range(K)]
    Alpha_K = np.random.rand(1, K)

    # Xi_DK and Alpha_D are the auxiliary variable related to the lower bound used for q(eta_d)
    Xi_DK = [np.random.rand(1, K) for i in range(D)]
    Alpha_D = np.random.rand(1, D)

    return Z, Eta, A, Rho, U_prime, U, Xi_KW, Alpha_K, Xi_DK, Alpha_D

def load_documents(word_emb_file, corpus_file):
    filein = open(word_emb_file, 'r')
    filein.readline()
    index = 0
    for line in filein:
        vals = line.strip().split()
        # build the word2idx dictionary
        word2idx[vals[0]] = index
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


# def gen_normalparams(dim):
#     mu_tmp = np.random.rand(1, dim)
#     Sigma_tmp = np.diag(np.random.rand(1, dim))
#     return dict(mu = mu_tmp, Sigma = Sigma_tmp)


def run():
	word_emb_file = '???'
	corpus_file = '???'
	l = 1
	c = 1
	kappa = 1
	beta = 1
	gamma = 1
	load_documents(word_emb_file, corpus_file)
    (Z, Eta, A, Rho, U_prime, U, Xi_KW, Alpha_K, Xi_DK, Alpha_D) = init_vars()
    # Yuxing Zhang TODO
    while true: # while not converge
        # TODO sample a batch of document B
        # TODO 
        for d in B:
            for w_dn in d:
                # TODO update q(z_dn) by Eq.7 
            # update Eta_d
<<<<<<< HEAD
            Eta[d] = update.update_eta(Eta[d], Xi_DK[d], Alpha_D[d], gamma, U, A[d])
=======
            Eta[d] = update.update_eta(Eta[d], Xi_DK[d], Alpha_D[d], gamma, U, A[d], q_Z[d])
>>>>>>> a553da0b8f8eda9c585ff7edae1616b2beff2325
            # update A_d
            A[d] = update.update_a(A[d], c, gamma, U, Eta[d])
            if certain_interval:
                # TODO update auxiliary variables ksi_d and alpha_d by Eq 2 and Eq 3


        # Tianshu Ren starts here TODO
        for k in K:
            # TODO update zeta_k_tild by Eq.8
            # TODO update u_k_tild by Eq.10
            # TODO update u_prime_k_tild by Eq.9
            if certain_interval():
                # TODO update xi_k by Eq. 5
                # TODO update alpha_k by Eq. 6

        if converge:
        	break

if __name__ == "__main__":
    run()
