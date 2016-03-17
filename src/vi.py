import sys
import numpy as np
import update
from sklearn.preprocessing import normalize

# Input: 
# 	Documents {B_1,...,B_D}
# Output:
# 	Variational parameters {u, u_prime, rho, alpha, eta, z}	
# Variables:
# 	word_dim:	Dimension of word embedding
# 	doc_dim:	Dimension of document embedding

def initialize_variables():
	# initialize Z s.t. Z_dn is a vector of size K as parameters for a categorical distribution
	Z = list()
	for d in range(D):
		Z.append([normalize(np.random.uniform(0,1,(1, K)), 'l1') for i in range(N[d])])
	# initialize Eta s.t. Eta_d contains two fields "mu" (1 * K) and "Sigma" (K*K) that specifies a multivariate gaussian
	Eta = list()
	for d in range(D):
		mu_tmp = np.random.rand(1, K)
		Sigma_tmp = np.diag(np.random.rand(1, K))
		Eta.append(dict(mu = mu_tmp, Sigma = Sigma_tmp))
	# initialize A s.t. A_d contains two fields "mu" (1 * doc_dim) and "Sigma" (doc_dim * doc_dim) that specifies a multivariate gaussian
	A = list()
	for d in range(D):
		mu_tmp = np.random.rand(1, doc_dim)
		Sigma_tmp = np.diag(np.random.rand(1, doc_dim))
		A.append(dict(mu = mu_tmp, Sigma = Sigma_tmp))
	# initialize Rho s.t. Rho_k contains two fields "mu" (1 * V) and "Sigma" (V * V) that specifies a multivariate gaussian
	Rho = list()
	for k in range(K):
		mu_tmp = np.random.rand(1, V)
		Sigma_tmp = np.diag(np.random.rand(1, V))
		Rho.append(dict(mu = mu_tmp, Sigma = Sigma_tmp))
	# initialize 
	U_prime = list()
def load_documents(filein):
	return B

def run():
	B = load_documents(filepath)
	initialize_variables()
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
