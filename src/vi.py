import sys
import numpy as np

# Input: 
# 	Documents {B_1,...,B_D}
# Output:
# 	Variational parameters {u, u_prime, rho, alpha, eta, z}	
# Variables:
# 	word_dim:	Dimension of word embedding
# 	doc_dim:	Dimension of document embedding

def initialize_variables():


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
