import sys
import nympy as np
from sklearn.preprocessing import normalize

def mylambda(xi):
	# Helper function on Page 5 under Eq. 1
	return 1 / (2 * xi) * (1 / (1 + exp(- xi)) - 0.5)

def update_q_z(zvec, d, n, K, V, mu_d, Rho, word_idx, xi, alpha):
	# update the vector z_dn of length K from Eq. 7
	# q(z_dn) is a multinomial distribution with q(z_dn=k) = z_dn(k)

	for k in range(K):
		E1 = mu_d(z)
		tmp1 = 0
		tmp2 = 0
		for w in V:
			tmp1 += mylambda(xi[z][w]) * (Rho[z]['Sigma'][w] ** 2 + Rho[z]['mu'][w] ** 2)
			tmp2 += xi[z][w] / 2
		E2 = Rho[z]['mu'][word_idx] + alpha[z](V / 2 - 1) - tmp1 - (0.5 - 2 * alpha[z] * 
			mylambda(xi[z][w])) * Rho[z]['mu'][w] + tmp2 - mylambda(xi[z][w]) * 
			(alpha[z] ** 2 - xi[z][w] ** 2) - np.log(1 + exp(xi[z][w]))
		zvec[z] = exp(E1 + E2)

	return normalize(zvec)

def update_mu(muvec, d,)