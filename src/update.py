import numpy as np
from sklearn.preprocessing import normalize
from numpy.linalg import inv


def mylambda(xi):
	# Helper function on Page 5 under Eq. 1
	return 1 / (2 * xi) * (1 / (1 + np.exp(- xi)) - 0.5)

def update_z(Z, d, n, K, V, mu_d, Rho, word_idx, xi, alpha):
	# update the vector z_dn of length K from Eq. 7
	# q(z_dn) is a multinomial distribution with q(z_dn=k) = z_dn(k)

	for z in range(K):
		E1 = mu_d(z)
		tmp = 0
		for w in V:
			tmp += mylambda(xi[z][w]) * (Rho['Sigma'][z][w] ** 2 + Rho['mu'][z][w] ** 2) - (1/2 - 2 * alpha[z] * mylambda(xi[z][w])) * Rho['mu'][z][w] \
					+ xi[z][w] / 2 - mylambda(xi[z][w]) * (alpha[z] ** 2 - xi[z][w] ** 2) - np.log(1 + np.exp(xi[z][w]))
		E2 = Rho['mu'][z][word_idx] + alpha[z](V / 2 - 1) - tmp
		Z[d][n][z] = np.exp(E1 + E2)
	Z[d][n] = normalize(Z[d][n])

def update_eta(Eta_d, Xi_DK_d, Alpha_D_d, gamma, U, A_d, q_Z_d):
	for k in range(K):
		Eta_d['Sigma'][k, k] = 1 / (gamma - 2 * mylambda(Xi_DK_d[k]))
		tmp = 0
		for w in range(V):
			tmp += q_Z_d[n][k] #??? how to reference ???
		Eta_d['mu'][k] = gamma * np.dot(U[k]['mu'].transpose(), A_d) + 2 * Alpha_D_d * mylambda(Xi_DK_d[k]) - 0.5 + tmp
		Eta_d['mu'][k] *= Eta_d['Sigma'][k, k]

	return Eta_d

def update_a(A_d, c, gamma, U, Eta_d):
	tmp1 = 0
	tmp2 = 0
	for k in range(K):
		tmp1 += np.dot(U[k]['mu'], U[k]['mu'])
		tmp2 += Eta_d['mu'][k] * U[k]['mu']
	A_d['Sigma'] = inv(gamma * tmp1 + gamma * K * U[k]['Sigma'] + c * np.identity(doc_dim))
	A_d['mu'] = gamma * A_d['Sigma'] * tmp2

	return A_d

def update_rho(Rho_k, k, q_Z, beta, word_emb, U_prime_k, Alpha_K_k, Xi_KW_k):
	for w in range(V):
		c_kw = 0
		m_k = 0
		for d in range(D):
			for n in range(N[d]):
				if W[d][n] == idx2word[w]:
					c_kw += q_Z[d][n][k]
				m_k += q_Z[d][n][k]
		Rho_k['Sigma'][w, w] = 1 / (beta + 2 * m_k * mylambda(Xi_KW_k[w]))
		Rho_k['mu'][w] = beta * np.dot(word_emb[w].transpose(), U_prime_k) + c_kw - m_k * (0.5 - 2 * Alpha_K_k * mylambda(Xi_KW_k))
		Rho_k['mu'][w] *= Rho_k['Sigma'][w, w]

	return Rho_k

def update_u_prime()
