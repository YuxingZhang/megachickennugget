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

def update_eta(d, Eta, Xi_DK, Alpha_D, gamma, U, A, q_Z):
	for k in range(K):
		Eta['Sigma'][d][k] = 1 / (gamma - 2 * mylambda(Xi_DK[d][k]))
		tmp = 0
		for n in range(N[d]):
			tmp += q_Z[d][n][k] #??? how to reference ???
		Eta['mu'][d][k] = gamma * np.dot(U[k]['mu'].transpose(), A[d]) + 2 * Alpha_D[d] * mylambda(Xi_DK[d][k]) - 0.5 + tmp
		Eta['mu'][d][k] *= Eta['Sigma'][d][k]

	return Eta

def update_a(d, A, c, gamma, U, Eta):
	# update Sigma only for the first document
	if d == 0:
		tmp1 = 0
		tmp2 = 0
		for k in range(K):
			tmp1 += np.dot(U['mu'][k].transpose(), U['mu'][k])
			tmp2 += Eta['mu'][d][k] * U['mu'][k]
		A['Sigma'] = inv(gamma * tmp1 + gamma * K * U[k]['Sigma'] + c * np.identity(doc_dim))
		A['mu'][d] = gamma * A['Sigma'] * tmp2
	else:
		tmp2 = 0
		for k in range(K):
			tmp2 += Eta['mu'][d][k] * U['mu'][k]
		A['mu'][d] = gamma * A['Sigma'] * tmp2

	return A

def update_rho(k, Rho, q_Z, beta, word_emb, U_prime, Alpha_K, Xi_KW):
	for w in range(V):
		c_kw = 0
		m_k = 0
		for d in range(D):
			for n in range(N[d]):
				if W[d][n] == idx2word[w]:
					c_kw += q_Z[d][n][k]
				m_k += q_Z[d][n][k]
		Rho['Sigma'][k][w] = 1 / (beta + 2 * m_k * mylambda(Xi_KW[k][w]))
		Rho['mu'][k][w] = beta * np.dot(word_emb[w].transpose(), U_prime[k]) + c_kw - m_k * (0.5 - 2 * Alpha_K[k] * mylambda(Xi_KW[k][w]))
		Rho['mu'][k][w] *= Rho['Sigma'][k][w]

	return Rho

def compute_u_prime_sigma(U_prime, beta, l, word_emb):
	tmp = 0
	for w in range(V):
		tmp += np.dot(word_emb[w], word_emb[w].transpose())
	U_prime['Sigma'] = inv(l * np.identity(word_dim) + beta * tmp)

	return U_prime

def update_u_prime(k, U_prime, beta, word_emb, Rho):
	tmp = 0
	for w in range(V):
		tmp += word_emb[w] * Rho['mu'][k][w]
	U_prime['mu'] = beta * U_prime['Sigma'] * tmp

	return U_prime






























