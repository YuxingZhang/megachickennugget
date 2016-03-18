import sys
import nympy as np
import sklearn
from sklearn.preprocessing import normalize
from numpy.linalg import inv

def mylambda(xi):
	# Helper function on Page 5 under Eq. 1
	return 1 / (2 * xi) * (1 / (1 + exp(- xi)) - 0.5)

def update_q_z(Z_dn, d, n, K, V, mu_d, Rho, word_idx, xi, alpha):
	# update the vector z_dn of length K from Eq. 7
	# q(z_dn) is a multinomial distribution with q(z_dn=k) = z_dn(k)

	for z in range(K):
		E1 = mu_d(z)
		tmp = 0
		# tmp2 = 0
		for w in V:
			tmp += mylambda(xi[z][w]) * (Rho[z]['Sigma'][w] ** 2 + Rho[z]['mu'][w] ** 2) - (1/2 - 2 * alpha[z] * mylambda(xi[z][w])) * Rho[z]['mu'][w] \
				   + xi[z][w] / 2 - mylambda(xi[z][w]) * (alpha[z] ** 2 - xi[z][w] ** 2) - np.log(1 + np.exp(xi[z][w]))
			# tmp2 += xi[z][w] / 2
		E2 = Rho[z]['mu'][word_idx] + alpha[z](V / 2 - 1) - tmp
		Z_dn[z] = np.exp(E1 + E2)
		Z_dn = normalize(Z_dn)
	#
	# return normalize(zvec)

def update_eta(Eta_d, Xi_DK_d, Alpha_D_d, gamma, U, A_d):
	for k in range(K):
		Eta_d['Sigma'][k] = 1 / (gamma - 2 * mylambda(Xi_DK_d[k]))
		tmp = 0
		for w in range(V):
			tmp += q_Z[d][n][k] #??? how to reference ???
		Eta_d['mu'][k, k] = gamma * np.dot(U[k]['mu'].transpose(), A_d) + 2 * Alpha_D_d * mylambda(Xi_DK_d[k]) - 0.5 + tmp
		Eta_d['mu'][k, k] *= Eta_d['Sigma'][k]

		return Eta_d

def update_a(A_d, c, gamma, U, Eta_d):
	tmp1 = 0
	tmp2 = 0
	for k in range(K):
		tmp1 += np.dot(U[k]['mu'], U[k]['mu'])
		tmp2 += Eta_d['mu'][k] * U[k]['mu']
	A_d['Sigma'] = inv(gamma * tmp + gamma * K * U[k]['Sigma'] + c * np.identity(doc_dim))
	A_d['mu'] = gamma * A_d['Sigma'] * tmp2

	return A_d