import sys
import numpy as np
import update
from sklearn.preprocessing import normalize
import warnings

# Input: 
#   Documents {B_1,...,B_D}
# Output:
#   Variational parameters {u, u_prime, rho, alpha, eta, z} 
# Variables:
#   W:          Words in each documents, W[d][n] is the n-th word in the d-th doc
#   N:          A list where N[d] = Number of words in document d
#   K:          Total number of topics
#   V:          Size of the vocabulary
#   D:          Total number of documents
#   word_dim:           Dimension of word embedding space
#   doc_dim:            Dimension of document embedding space
#   word_emb:           Word-embedding results for each word
#   word2idx:           Index of each word in the word_emb vector

def init_vars(D, K, V, N, doc_dim, word_dim):
    # initialize Z s.t. Z_dn is a vector of size K as parameters for a categorical distribution
    # Z is the variational distribution of q(z_dn), q(z_dn = k) = Z(d, n, k)
    Z = list()
    for d in range(D):
        Z.append([normalize(np.random.uniform(0, 1, K), 'l1') for i in range(N[d])])

    # initialize Eta with parameters Sigma (D * K) and mu (D * K) that defines a multivariate Gaussian distribution
    # Eta_{dk} ~ Normal(mu[d][k], Sigma[d][k])
    Eta = dict(Sigma = [np.random.rand(K) for d in range(D)], mu = [np.random.rand(K) for d in range(D)])

    # initialize A with parameters Sigma (doc_dim * doc_dim) and mu (D * doc_dim) such that
    # A_d ~ Normal(mu[d], Sigma)
    A = dict(Sigma = np.diag(np.random.rand(doc_dim)), mu = [np.random.rand(doc_dim) for d in range(D)])

    # initialize Rho with parameters Sigma (K * V) and mu (K * V) that defines a multivariate Gaussian distribution
    # Rho_{kw} ~ Normal(mu[k][w], Sigma[k][w])
    Rho = dict(Sigma = [np.random.rand(V) for k in range(K)], mu = [np.random.rand(V) for k in range(K)])

    # initialize U_prime with parameters Sigma (word_dim * word_dim) and mu (K * word_dim) such that
    # U_prime_k ~ Normal(mu[k], Sigma)
    U_prime = dict(Sigma = np.diag(np.random.rand(word_dim)), mu = [np.random.rand(word_dim) for k in range(K)])

    # initialize U with parameters Sigma (doc_dim * doc_dim) and mu (K * doc_dim) such that
    # U_k ~ Normal(mu[k], Sigma)
    U = dict(Sigma = np.diag(np.random.rand(doc_dim)), mu = [np.random.rand(doc_dim) for k in range(K)])

    ''' Xi_KW and Alpha_K are the auxiliary variable related to the lower bound used for q(z_dn) and q(rho) '''
    Xi_KW = [np.random.rand(V) for i in range(K)]
    Alpha_K = np.random.rand(K)

    ''' Xi_DK and Alpha_D are the auxiliary variable related to the lower bound used for q(eta_d) '''
    Xi_DK = [np.random.rand(K) for i in range(D)]
    Alpha_D = np.random.rand(D)

    return Z, Eta, A, Rho, U_prime, U, Xi_KW, Alpha_K, Xi_DK, Alpha_D

def load_documents(word_emb_file, corpus_file):
    '''
    This function read and load the word embedding and the corpus
        word_emb: store the embedding of the words in the same order as the index
        word2idx: store the index of each word, using the order of the words in the vocabulary
        idx2word: can access the word from the index
        W: all the documents, the i-th word in the d-th document is W[d][i]
    '''
    W = list()
    N = list()
    word2idx = dict() # mapping from a word to it's index in the vocabulary
    idx2word = dict() # mapping from index of a word in the vocabulary to the word itself

    f = open(word_emb_file, 'r')
    first_line = f.readline().strip().split()
    vocabulary_size = first_line[0]
    embedding_dim = first_line[1]

    word_emb = np.empty([int(vocabulary_size), int(embedding_dim)]) # word embedding from word2vec

    dat = f.readlines()
    f.close()
    index = 0
    for line in dat:
        vec = np.asarray(line.strip().split())

        # build the word2idx dictionary and the idx2word dictionary
        word2idx[vec[0]] = index
        idx2word[index] = vec[0]

        # store the word-embedding results
        word_emb[index] = vec[1: ]
        index += 1

    # read in the corpus file
    f = open(corpus_file, 'r')
    dat = f.readlines()
    f.close()
    for doc in dat:
        words = doc.strip().split()
        W.append(words)
        N.append(len(words))

    word_emb = np.asarray(word_emb) # convert to numpy array
    return word_emb, word2idx, idx2word, W, N

def run():
#   W:                  Words in each documents, W[d][n] is the n-th word in the d-th doc
#   N:                  A list where N[d] = Number of words in document d
#   K:                  Total number of topics
#   V:                  Size of the vocabulary
#   D:                  Total number of documents
#   word_dim:                   Dimension of word embedding space
#   doc_dim:                    Dimension of document embedding space
#   word_emb:           Word-embedding results for each word
#   word2idx:           Index of each word in the word_emb vector
#       l, c, kappa, beta, gamma        Hyper parameters for the model

    # initialize all variables
    l = 1
    c = 1
    kappa = 1
    beta = 1
    gamma = 1

    K = 10 # number of topics
    V = 0 # 
    D = 0 #
    word_dim = 200
    doc_dim = 100 # embedding space dimension of document

    word_emb_input = '../vectors.txt' # TODO
    corpus_input = '../new_corpus.txt' # TODO
    (word_emb, word2idx, idx2word, W, N) = load_documents(word_emb_input, corpus_input)

    # setting the vocabulary size
    V = len(word2idx)
    D = len(W)

    (Z, Eta, A, Rho, U_prime, U, Xi_KW, Alpha_K, Xi_DK, Alpha_D) = init_vars(D, K, V, N, doc_dim, word_dim)

    random_idx = np.random.permutation(len(W))
    batch_size = 20
    eps = 0.01
    number_of_batch = int((len(random_idx) + batch_size - 1) / batch_size)
    current_batch = number_of_batch

    iter = 0
    MAX_ITER = 100
    while iter < MAX_ITER:
        # VI step to update variational parameters
        while True: # while not converge
            # TODO precompute Sigma^{(u')*} by Eq. 9
            update.compute_u_prime_sigma(U_prime, word_emb, beta, l, word_dim, V)
            print current_batch
            has_converge = True
            # randomly sample a batch of document B
            current_batch -= 1
            if current_batch < 0:
                current_batch += number_of_batch
            B = random_idx[current_batch * batch_size : (current_batch + 1) * batch_size]
            # update local distribution
            for d in B:
                for n in range(N[d]):
                    cvg = update.update_z(d, n, Z, Eta, Rho, Xi_KW, Alpha_K, W, word2idx, K, V, eps)
                    if not cvg:
                        has_converge = False
                # update Eta
                cvg = update.update_eta(d, Eta, Xi_DK, Alpha_D, U, A, Z, gamma, N, K, eps)
                if not cvg:
                    has_converge = False
                # update A
                cvg = update.update_a(d, A, U, Eta, c, gamma, doc_dim, K, eps)
                if not cvg:
                    has_converge = False
                update.update_auxiliary(d, Alpha_D, Xi_DK, Eta, K)  # update the auxiliary vars using in q(eta)
            # update global distributions
            for k in range(K):
                # update Rho
                cvg = update.update_rho(k, Rho, Z, U_prime, Alpha_K, Xi_KW, word_emb, W, idx2word, beta, D, N, V, eps)
                if not cvg:
                    has_converge = False
                # update U
                cvg = update.update_u(k, U, A, Eta, kappa, gamma, doc_dim, D, eps)
                if not cvg:
                    has_converge = False
                # update U_prime
                cvg = update.update_u_prime(k, U_prime, Rho, word_emb, beta, V, eps)
                if not cvg:
                    has_converge = False
                update.update_auxiliary(k, Alpha_K, Xi_KW, Rho, V)  # update the auxiliary vars using in q(z_dn) and q(rho)

            if has_converge:
                break

        # Update parameters in the original distribution p
        l = update.update_l(U_prime, word_dim, K)
        kappa = update.update_kappa(U, doc_dim, K)
        c = update.update_c(A, doc_dim, D)

        iter += 1

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    run()

