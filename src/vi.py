import sys
import numpy as np
import update
from sklearn.preprocessing import normalize

# Input: 
# 	Documents {B_1,...,B_D}
# Output:
# 	Variational parameters {u, u_prime, rho, alpha, eta, z}	
# Variables:
#	K:			Number of topics
#	V:			Size of vocabulary
#	D:			Number of documents
#	N:			A list where N[i] = Number of words in document i
# 	word_dim:	Dimension of word embedding
# 	doc_dim:	Dimension of document embedding

K = 10 # number of topics
doc_dim = 100 # embedding space dimension of document
V = 10000 # 
word_dim = 100
D = 20 #

def init_vars(D, K, N, V, doc_dim, word_dim):
	# initialize Z s.t. Z_dn is a vector of size K as parameters for a categorical distribution
        # Z is the variational distribution of q(z_dn), q(z_dn = k) = Z(d, n, k)
	Z = list()
	for d in range(D):
		Z.append([normalize(np.random.uniform(0, 1, (1, K)), 'l1') for i in range(N[d])])
	# initialize Eta s.t. Eta_d contains two fields "mu" (1 * K) and "Sigma" (K*K) that specifies a multivariate gaussian
	Eta = list()
	for d in range(D):
		Eta.append(gen_normalparams(K))
	# initialize A s.t. A_d contains two fields "mu" (1 * doc_dim) and "Sigma" (doc_dim * doc_dim) that specifies a multivariate gaussian
	A = list()
	for d in range(D):
		A.append(gen_normalparams(doc_dim))
	# initialize Rho s.t. Rho_k contains two fields "mu" (1 * V) and "Sigma" (V * V) that specifies a multivariate gaussian
	Rho = list()
	for k in range(K):
		Rho.append(gen_normalparams(V))
	# initialize U_prime s.t. U_prime_k contains two fields "mu" (1 * word_dim) and "Sigma" (word_dim * word_dim) that specifies a multivariate gaussian
	U_prime = list()
	for k in range(K):
		U_prime.append(gen_normalparams(word_dim))
	# initialize U s.t. U_k contains two fields "mu" (1 * doc_dim) and "Sigma" (doc_dim, doc_dim)
	U = list()
	for k in range(K):
		U.append(gen_normalparams(doc_dim))
	XiK = [np.random.random((1, V))	for i in range(K)]
	AlphaK = np.random.random((1, K))
	XiD = 1 #TODO
	AlphaD = 1 #TODO

	return Z, Eta, A, Rho, U_prime, U, XiK, AlphaK, XiD, AlphaD

def load_documents(word_embd_file, corpus_file):
	word_embd = list()
	dictionary = dict()
	filein = open(word_embd_file, 'r')
	filein.readline()
	index = 0
	for line in filein:
		vals = line.split()
		# build the dictionary
		dictionary[vals[0]] = index
		# store the word-embedding results
		word_embd.append(vals[1: ])
		index += 1
	filein.close()

	# read in the corpus file
	W = list()
	N = list()
	filein = open(corpus_file, 'r')
	for doc in filein:
		words = doc.split()
		W.append(words)
		N.append(len(words))

	filein.close()

	return dictionary, word_embd, W

def gen_normalparams(dim):
	mu_tmp = np.random.rand(1, dim)
	Sigma_tmp = np.diag(np.random.rand(1, dim))
	return dict(mu = mu_tmp, Sigma = Sigma_tmp)

def run():
	B = load_documents(file_path)
	(Z, Eta, A, Rho, U_prime, U, XiK, AlphaK, XiD, AlphaD) = init_vars(D, K, N, V, doc_dim, word_dim)
    # Yuxing Zhang TODO
    while true: # while not converge
        # TODO sample a batch of document B
        # TODO 
        for d in B:
            for w_dn in d:
                # TODO update q(z_dn) by Eq.7 
            # TODO update ~mu_d by Eq 上次最后推出来的
            # TODO update ~a_d by Eq 上次最后推出来的
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
    			


if __name__ == "__main__":
    run()
