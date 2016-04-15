#include <fstream>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <set>
#include <armadillo>
#include <algorithm>
using namespace std;
using namespace arma;
#include "load.cpp"

void load_files(string embedding, string corpus, double** word_embd, map<string, int>& word2idx, map<int, string>& idx2word, vector<vector<string> >& W, vector<int>& N);

int main() {
    // Reading input files, including the corpus and the embedding
    string emb_file = "../vectors.txt";
    string corpus_file = "../new_corpus.txt";

    double** word_embedding;    // V * dim, each line is a vector of double
    map<string, int> word2idx;  // V
    map<int, string> idx2word;  // V
    vector< vector<string> > W; // W[d][n] is w_dn
    vector<int> N; // N[d] is the length of document d
    load_files(emb_file, corpus_file, word_embedding, word2idx, idx2word, W, N);

    // some parameters
    const int V = idx2word.size(); // vocabulary size
    const int D = W.size(); // number of documents
    const int K = 10; // number of topics
    const int WORD_DIM = 200; // dimension of word embedding
    const int DOC_DIM = 100;// dimension of document embedding

    // model parameters, changed in the M step
    double l = 1.0; 
    double c = 1.0;
    double kappa = 1.0;
    double beta = 1.0;
    double gamma = 1.0;

    // initialization
    vector<mat> z; // each element is a n_d * K matrix 
    for (int i = 0; i < D; i++) {
        mat tmp(N[i], K, fill::randu);
        normalise(tmp, 1, 1);
        z.push_back(tmp);
    }

    // flag is diagonal or not
    mat eta_m(D, K, fill::randu); // mean of eta
    mat eta_s(D, K); // sigma of eta

    mat a_m(D, DOC_DIM, fill::randu);
    mat a_s = diagmat(vec(DOC_DIM, fill::randu)); // all a_d share the same matrix

    mat rho_m(K, V, fill::randu); // mean of rho
    mat rho_s(K, V, fill::randu); // sigma of rho

    mat up_m(K, WORD_DIM, fill::randu); // mean of u prime
    mat up_s = diagmat(vec(WORD_DIM, fill::randu)); // sigma of u prime, shared

    mat u_m(K, DOC_DIM, fill::randu); // mean of u
    mat u_s = diagmat(vec(DOC_DIM, fill::randu)); // sigma of u, shared

    mat xi_KW(K, V, fill::randu); // used in lower bound of z_dn and rho
    vec alpha_K(K, fill::randu);

    mat xi_DK(D, K, fill::randu); // used in lower bound of eta_d
    vec alpha_D(D, fill::randu);

    // train for each batch
    vector<int> random_index;
    for (int i = 0; i < W.size(); i++) {
        random_index.push_back(i);
    }
    random_shuffle(random_index.begin(), random_index.end());
    const int BATCH_SIZE = 20;
    const double EPS = 0.01;
    int num_of_batch = (int)((random_index.size() + BATCH_SIZE 
            - 1) / BATCH_SIZE);
    int cur_batch = num_of_batch;

    int iteration = 0;
    while (true) {
        iteration++;

        while (true) {
            //ComputeUpSigma(up_s, word_embedding, beta, l, WORD_DIM, V);

            bool has_converge = true;
            cur_batch--;
            if (cur_batch < 0) {
                cur_batch += num_of_batch;
            }

            /*
            // create a batch index set
            set<int> idx_set;
            for (int i = cur_batch * BATCH_SIZE; i < (cur_batch + 1) * BATCH_SIZE && i < random_index.size(); i++) {
                idx_set.insert(random_index[i]);
            }

            for (set<int>::iterator d = idx_set.begin(); d != idx_set.end(); d++) {
                for (int n = 0; n < N[*d]; n++) {
                    if (!UpdateZ(*d, n, z, eta_m)) { has_converge = false; }
                }
                if (!UpdateEta(*d, eta_m, eta_s, xi_DK, alpha_D, u_m, a_m, z, gamma, N, K, EPS)) { has_converge = false; }
            }
            */
        }
    }
    return 0;
}
