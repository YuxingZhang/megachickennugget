import numpy as np
from sklearn.preprocessing import normalize
from numpy.linalg import inv

''' 
FUNCTION ARGUMENTS ORDER:
    function(index, parameters_to_change, parameters_to_use, global_parameters, CONSTANT)
'''

def lmd(xi):
    # Helper function on Page 5 under Eq. 1
    # Last checked Mar. 27 1:55pm
    return 1 / (2 * xi) * (1 / (1 + np.exp(- xi)) - 0.5)


def update_auxiliary(idx, Alpha, Xi, Var, Sidx):
    '''
    q(Var[idx]) = \prod_{i=1}^{Sidx} N(Var[idx][i]; Var['mu'][idx][i], Var['Sigma'][idx][i])

    :param idx: the index to update in Alpha and Xi
    :param Alpha: a vector s.t. Alpha[idx] is a scalar that corresponds to the auxiliary variable used to update Var[idx]
    :param Xi: a matrix s.t. Xi[idx] is a vector that corresponds to the auxiliary variables used to update Var[idx]
    :param Var: the random variable, e.g. Rho, Eta, Z that uses the auxiliary variable
    :param Sidx: the index that is summed over in this update, i.e. sum_{i=1}^Sidx
    :return: null. Update in function
    '''
    tmp1 = 0
    tmp2 = 0
    for i in range(Sidx):
        tmp1 += lmd(Xi[idx][i]) * Var['mu'][i]
        tmp2 += lmd(Xi[idx][i])
        Xi[idx] = np.sqrt(Var['Sigma'][i] + Var['mu'][i] ** 2 - 2 * Alpha[idx] * Var['mu'][i] + Alpha[idx] ** 2)
    Alpha[idx] = (0.5 * (Sidx / 2 - 1) + tmp1) / tmp2


def update_z(d, n, Z, Eta, Rho, Xi_KW, Alpha_K, W, word2idx, K, V): 
    # Update the vector q(z_dn) of length K from Eq. 7
    # q(z_dn) is a multinomial distribution with q(z_dn=k) = Z[d][n][k]
    for k in range(K):
        E1 = Eta[d][k]  # First expectation term

        tmp = 0
        for w in V:
            tmp +=  - lmd(Xi_KW[k][w]) * (Rho['Sigma'][k][w] + Rho['mu'][k][w] ** 2) \
                    - (0.5 - 2 * Alpha_K[k] * lmd(Xi_KW[k][w])) * Rho['mu'][k][w] \
                    + Xi_KW[k][w] / 2 \
                    - lmd(Xi_KW[k][w]) * (Alpha_K[k] ** 2 - Xi_KW[k][w] ** 2) \
                    - np.log(1 + np.exp(Xi_KW[k][w]))

        w_dn = W[d][n]
        E2 = Rho['mu'][k][word2idx[w_dn]] + Alpha_K[k] * (V / 2 - 1) + tmp  # Second expectation term
        Z[d][n][k] = np.exp(E1 + E2)
    Z[d][n] = normalize(Z[d][n])


def update_eta(d, Eta, Xi_DK, Alpha_D, U, A, Z, gamma, N, K):
    # Update q(eta_d) by Eq. (11) and (12)
    # Last checked Mar. 27 2:31pm
    for k in range(K):
        Eta['Sigma'][d][k] = 1 / (gamma - 2 * lmd(Xi_DK[d][k]))
        tmp = 0
        for n in range(N[d]):
            tmp += Z[d][n][k]
        Eta['mu'][d][k] = gamma * np.dot(U['mu'][k].transpose(), A['mu'][d]) + 2 * Alpha_D[d] * lmd(Xi_DK[d][k]) - 0.5 + tmp
        Eta['mu'][d][k] *= Eta['Sigma'][d][k]


def update_a(d, A, U, Eta, c, gamma, doc_dim, K):
    # Update Sigma only for the first document
    # Last checked Mar. 27 3:06pm
    if d == 0:
        tmp1 = 0
        tmp2 = 0
        for k in range(K):
            tmp1 += np.dot(U['mu'][k], U['mu'][k].transpose())
            tmp2 += Eta['mu'][d][k] * U['mu'][k]
        A['Sigma'] = inv(gamma * tmp1 + gamma * K * U['Sigma'] + c * np.identity(doc_dim))
        A['mu'][d] = gamma * np.dot(A['Sigma'], tmp2)
    else:
        tmp2 = 0
        for k in range(K):
            tmp2 += Eta['mu'][d][k] * U['mu'][k]
        A['mu'][d] = gamma * np.dot(A['Sigma'], tmp2)


def update_rho(k, Rho, Z, U_prime, Alpha_K, Xi_KW, word_emb, W, idx2word, beta, D, N, V):
    # Update parameters for q(rho_k) by Eq. 2 and 3
    # Last checked Mar. 27 1:53pm
    for w in range(V):
        c_kw = 0
        m_k = 0
        for d in range(D):
            for n in range(N[d]):
                if W[d][n] == idx2word[w]:
                    c_kw += Z[d][n][k]
                m_k += Z[d][n][k]
        Rho['Sigma'][k][w] = 1 / (beta + 2 * m_k * lmd(Xi_KW[k][w]))
        Rho['mu'][k][w] = beta * np.dot(word_emb[w].transpose(), U_prime['mu'][k]) + c_kw - m_k * (0.5 - 2 * Alpha_K[k] * lmd(Xi_KW[k][w])) # word_emb[w].transpose()?
        Rho['mu'][k][w] *= Rho['Sigma'][k][w]


def update_u_prime(k, U_prime, Rho, word_emb, beta, V):
    # Update U'[Sigma] only for the first topic
    # Last checked Mar. 27 4:38pm
    tmp = 0
    for w in range(V):
        tmp += word_emb[w] * Rho['mu'][k][w]
    U_prime['mu'][k] = beta * np.dot(U_prime['Sigma'], tmp)


def update_u(k, U, A, Eta, kappa, gamma, doc_dim, D):
    # Update parameters for q(u) by Eq. (7) and (8)
    # Last checked Mar. 27 3:34pm
    if k == 0:
        tmp1 = 0
        tmp2 = 0
        for d in range(D):
            tmp1 += np.dot(A['mu'][d], A['mu'][d].transpose())
            tmp2 += Eta['mu'][d][k] * A['mu'][d]
        U['Sigma'] = inv(kappa * np.identity(doc_dim) + gamma * D * A['Sigma'] + gamma * tmp1)
        U['mu'][k] = gamma * np.dot(U['Sigma'], tmp2)
    else:
        tmp2 = 0
        for d in range(D):
            tmp2 += Eta['mu'][d][k] * A['mu'][d]
        U['mu'][k] = gamma * np.dot(U['Sigma'], tmp2)


def compute_u_prime_sigma(U_prime, word_emb, beta, l, word_dim, V):
    tmp = 0
    for w in range(V):
        tmp += np.dot(word_emb[w], word_emb[w].transpose())
    U_prime['Sigma'] = inv(l * np.identity(word_dim) + beta * tmp)


