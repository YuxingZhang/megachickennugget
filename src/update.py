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
    return 1.0 / (2.0 * xi) * (1.0 / (1.0 + np.exp(- xi)) - 0.5)


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
    '''
    print len(Xi)
    print Sidx
    print Xi[0].shape
    print Var['mu'][0].shape
    print Var['Sigma'][0].shape
    print "start loop"
    '''
    for i in range(Sidx):
        '''
        print "==========="
        print "i = " + str(i)
        print "idx = " + str(idx)
        print "Xi[idx][i] = " + str(Xi[idx][i])
        '''
        tmp2 += lmd(Xi[idx][i])
        tmp1 += lmd(Xi[idx][i]) * Var['mu'][idx][i]
        Xi[idx][i] = np.sqrt(Var['Sigma'][idx][i] + Var['mu'][idx][i] ** 2 - 2 * Alpha[idx] * Var['mu'][idx][i] + Alpha[idx] ** 2)
    Alpha[idx] = (0.5 * (Sidx / 2.0 - 1.0) + tmp1) / tmp2


def update_z(d, n, Z, Eta, Rho, Xi_KW, Alpha_K, W, word2idx, K, V, eps):
    # Update the vector q(z_dn) of length K from Eq. 7
    # q(z_dn) is a multinomial distribution with q(z_dn=k) = Z[d][n][k]

    converge = True
    z_dn_old = np.array(Z[d][n])

    # print "hello from the other side, n = " + str(n)

    for k in range(K):

        E1 = Eta['mu'][d][k]  # First expectation term

        tmp = 0.0
        for w in range(V):
            tmp +=  - lmd(Xi_KW[k][w]) * (Rho['Sigma'][k][w] + Rho['mu'][k][w] ** 2) \
                    - (0.5 - 2.0 * Alpha_K[k] * lmd(Xi_KW[k][w])) * Rho['mu'][k][w] \
                    + Xi_KW[k][w] / 2.0 \
                    - lmd(Xi_KW[k][w]) * (Alpha_K[k] ** 2 - Xi_KW[k][w] ** 2) \
                    - np.log(1.0 + np.exp(Xi_KW[k][w]))

        w_dn = W[d][n]
        # print tmp
        E2 = Rho['mu'][k][word2idx[w_dn]] + Alpha_K[k] * (V / 2.0 - 1.0) + tmp  # Second expectation term
        # print "E2" + str(E2)
        # print "E1" + str(E1)
        Z[d][n][k] = np.exp(E1 + E2)
    Z[d][n] = normalize(Z[d][n])[0]

    for k in range(K):
        if abs(Z[d][n][k] - z_dn_old[k]) / abs(z_dn_old[k]) > eps:
            converge = False
            break
    return converge


def update_eta(d, Eta, Xi_DK, Alpha_D, U, A, Z, gamma, N, K, eps):
    # Update q(eta_d) by Eq. (11) and (12)
    # Last checked Mar. 27 2:31pm

    converge = True
    mu_old = np.array(Eta['mu'][d])
    sig_old = np.array(Eta['Sigma'][d])

    for k in range(K):
        Eta['Sigma'][d][k] = 1.0 / (gamma + 2.0 * N[d] * lmd(Xi_DK[d][k]))
        tmp = 0.0
        for n in range(N[d]):
            tmp += Z[d][n][k]
        Eta['mu'][d][k] = gamma * np.dot(U['mu'][k].transpose(), A['mu'][d]) + N[d] * (2.0 * Alpha_D[d] * lmd(Xi_DK[d][k]) - 0.5) + tmp
        Eta['mu'][d][k] *= Eta['Sigma'][d][k]

    for k in range(K):
        if max(abs(Eta['mu'][d][k] - mu_old[k]) / abs(mu_old[k]), abs(Eta['Sigma'][d][k] - sig_old[k]) / abs(sig_old[k])) > eps:
            converge = False
            break
    return converge


def update_a(d, A, U, Eta, c, gamma, doc_dim, K, eps):
    # Update Sigma only for the first document
    # Last checked Mar. 27 3:06pm

    converge = True
    mu_old = np.array(A['mu'][d])
    sig_old = np.array(A['Sigma'])

    if d == 0:
        tmp1 = 0.0
        tmp2 = 0.0
        for k in range(K):
            tmp1 += np.outer(U['mu'][k], U['mu'][k])
            tmp2 += Eta['mu'][d][k] * U['mu'][k]
        A['Sigma'] = inv(gamma * tmp1 + gamma * K * U['Sigma'] + c * np.identity(doc_dim))
        A['mu'][d] = gamma * np.dot(A['Sigma'], tmp2)
    else:
        tmp2 = 0.0
        for k in range(K):
            tmp2 += Eta['mu'][d][k] * U['mu'][k]
        A['mu'][d] = gamma * np.dot(A['Sigma'], tmp2)

    for k in range(K):
        if abs(A['mu'][d][k] - mu_old[k]) / abs(mu_old[k]) > eps:
            converge = False
            break
    if converge:
        if (abs(A['Sigma'] - sig_old) / abs(sig_old)).max() > eps:
            converge = False
    return converge


def update_rho(k, Rho, Z, U_prime, Alpha_K, Xi_KW, word_emb, W, idx2word, beta, D, N, V, eps):
    # Update parameters for q(rho_k) by Eq. 2 and 3
    # Last checked Mar. 27 1:53pm

    converge = True
    mu_old = np.array(Rho['mu'][k])
    sig_old = np.array(Rho['Sigma'][k])

    for w in range(V):
        c_kw = 0.0
        m_k = 0.0
        for d in range(D):
            for n in range(N[d]):
                if W[d][n] == idx2word[w]:
                    c_kw += Z[d][n][k]
                m_k += Z[d][n][k]
        Rho['Sigma'][k][w] = 1.0 / (beta + 2.0 * m_k * lmd(Xi_KW[k][w]))
        Rho['mu'][k][w] = beta * np.dot(word_emb[w].transpose(), U_prime['mu'][k]) + c_kw - m_k * (0.5 - 2.0 * Alpha_K[k] * lmd(Xi_KW[k][w])) # word_emb[w].transpose()?
        Rho['mu'][k][w] *= Rho['Sigma'][k][w]

    for w in range(V):
        if max(abs(Rho['mu'][k][w] - mu_old[w]) / abs(mu_old[w]), abs(Rho['Sigma'][k][w] - sig_old[w]) / abs(sig_old[w])) > eps:
            converge = False
            break
    return converge


def update_u_prime(k, U_prime, Rho, word_emb, beta, V, eps):
    # Update U'['mu'] by Eq. (5)
    # Last checked Mar. 27 4:38pm

    converge = True
    mu_old = np.array(U_prime['mu'][k])

    tmp = 0.0
    for w in range(V):
        tmp += word_emb[w] * Rho['mu'][k][w]
    U_prime['mu'][k] = beta * np.dot(U_prime['Sigma'], tmp)

    for w in range(V):
        if abs(U_prime['mu'][k][w] - mu_old[w]) / abs(mu_old[w]) > eps:
            converge = False
            break
    return converge


def update_u(k, U, A, Eta, kappa, gamma, doc_dim, D, eps):
    # Update parameters for q(u) by Eq. (7) and (8)
    # Last checked Mar. 27 3:34pm
    converge = True
    mu_old = np.array(U['mu'][k])
    sig_old = np.array(U['Sigma'])

    if k == 0:
        tmp1 = 0.0
        tmp2 = 0.0
        for d in range(D):
            tmp1 += np.outer(A['mu'][d], A['mu'][d])
            tmp2 += Eta['mu'][d][k] * A['mu'][d]
        U['Sigma'] = inv(kappa * np.identity(doc_dim) + gamma * D * A['Sigma'] + gamma * tmp1)
        U['mu'][k] = gamma * np.dot(U['Sigma'], tmp2)
    else:
        tmp2 = 0
        for d in range(D):
            tmp2 += Eta['mu'][d][k] * A['mu'][d]
        U['mu'][k] = gamma * np.dot(U['Sigma'], tmp2)

    for d in range(D):
        if abs(U['mu'][k][d] - mu_old[d]) / abs(mu_old[d]) > eps:
            converge = False
            break
    if converge:
        if (abs(U['Sigma'] - sig_old) / abs(sig_old)).max() > eps:
            converge = False
    return converge

def compute_u_prime_sigma(U_prime, word_emb, beta, l, word_dim, V):
    # Update U'['Sigma'] only at the start of VI by Eq. (6)
    tmp = 0.0
    for w in range(V):
        tmp += np.outer(word_emb[w], word_emb[w].transpose())
    U_prime['Sigma'] = inv(l * np.identity(word_dim) + beta * tmp)

def update_l(U_prime, word_dim, K):
    tmp = 0.0
    for k in range(K):
        tmp += np.dot(U_prime['mu'][k].transpose(), U_prime['mu'][k])
    return K * word_dim / (tmp + K * np.trace(U_prime['Sigma']))

def update_kappa(U, doc_dim, K):
    tmp = 0.0
    for k in range(K):
        tmp += np.dot(U['mu'][k].transpose(), U['mu'][k])
    return K * doc_dim / (tmp + K * np.trace(U['Sigma']))

def update_c(A, doc_dim, D):
    tmp = 0.0
    for d in range(D):
        tmp += np.dot(A['mu'][d].transpose(), A['mu'][d])
    return D * doc_dim / (tmp + D * np.trace(A['Sigma']))

def update_beta(U_prime, word_emb, Rho, V, K):
    tmp = 0.0
    for k in range(K):
        for w in range(V):
            tmp += np.trace(np.dot(np.dot(U_prime['mu'][k], U_prime['mu'][k].transpose())
            + U_prime['Sigma'], np.dot(word_emb[w], word_emb[w].transpose()))) + Rho['mu'][k][w] ** 2
            + Rho['Sigma'][k][w] - 2.0 * np.dot(word_emb[w].transpose(), U_prime['mu'][k]) * Rho['mu'][k][w]
    return K * V / tmp

def update_gamma(Eta, A, U, D, K):
    tmp = 0.0
    for d in range(D):
        for k in range(K):
            tmp += Eta['mu'][d][k] ** 2 + Eta['Sigma'][d][k] - 2.0 * Eta['mu'][d][k] * np.dot(U['mu'][k].transpose(),
                A['mu'][d]) + np.trace(np.dot(np.dot(A['mu'][d], A['mu'][d].transpose()) + A['Sigma'],
                np.dot(U['mu'][k], U['mu'][k].transpose()) + U['Sigma']))
    return D * K / tmp

