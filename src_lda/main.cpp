#include <fstream>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <set>
#include <armadillo>
#include <algorithm>
#include <cmath>
using namespace std;
using namespace arma;
#include "digamma.cpp"
#include "load.cpp"
#include "update.cpp"
#include "opt_beta.cpp"
#include "sample_eta.cpp"

void load_files(string embedding, string corpus, map<string, int>& word2idx, map<int, string>& idx2word, vector<vector<string> >& W, vector<int>& N);

int main() {
    // Reading input files, including the corpus and the embedding
    string emb_file = "../abs_vectors.txt";
    string corpus_file = "../abs_corpus.txt";
    //arma_rng::set_seed_random();
    map<string, int> word2idx;  // V
    map<int, string> idx2word;  // V
    vector< vector<string> > W; // W[d][n] is w_dn
    vector<int> N; // N[d] is the length of document d
    load_files(emb_file, corpus_file, word2idx, idx2word, W, N); // V * dim, each line is a vector of double

    // some parameters
    const int V = idx2word.size(); // vocabulary size
    const int D = W.size(); // number of documents
    const int K = 5; // number of topics

    vec beta(V, fill::ones);
    vec alpha(K, fill::ones);

    // initialization
    vector<mat> z; // each element is a n_d * K matrix 
    for (int i = 0; i < D; i++) {
        mat tmp(N[i], K, fill::randu);
        tmp = normalise(tmp, 1, 1);
        z.push_back(tmp);
    }
    mat gamma(D, K, fill::randu); 
    mat lambda(K, V, fill::randu); 

    // train for each batch
    vector<int> random_index;
    for (size_t i = 0; i < W.size(); i++) {
        random_index.push_back(i);
    }
    random_shuffle(random_index.begin(), random_index.end());
    const int BATCH_SIZE = 87;
    const double EPS = 0.1;
    int num_of_batch = (int)((random_index.size() + BATCH_SIZE - 1) / BATCH_SIZE);
    int cur_batch = num_of_batch;

    cout << "vocabulary size = " << V << endl;

    int iteration = 0;
    int MAX_ITER = 10;
    int aux_iter = 10;
    double elbo, prev_elbo;

    while (iteration < MAX_ITER) {
        iteration++;

        int inner_iteration = 0;
        // E-step: Variational Inference
        while (inner_iteration < 2 * MAX_ITER) {
            prev_elbo = elbo;
            elbo = 0.0;
            inner_iteration++;
            cout << inner_iteration << endl;
            bool has_converge = true;
            cur_batch--;
            if (cur_batch < 0) {
                cur_batch += num_of_batch;
            }

            // create a batch index set
            set<int> idx_set;
            for (int i = cur_batch * BATCH_SIZE; i < (cur_batch + 1) * BATCH_SIZE && i < (int)random_index.size(); i++) {
                idx_set.insert(random_index[i]);
            }

            // update Z
            elbo += UpdateZ(idx_set, N, z, gamma, lambda, W, word2idx, K, V);

            // update_Gamma            
            for (set<int>::iterator d = idx_set.begin(); d != idx_set.end(); d++) {
                elbo += UpdateGamma(*d, gamma, z, N, K, alpha);
            }

            // Update lambda
            for (int k = 0; k < K; k++) {
                elbo += UpdateLambda(k, lambda, z, W, word2idx, beta, D, N, V);
            }

            cout << "iteration finished" << endl;
            if (abs(prev_elbo - elbo) / abs(prev_elbo) < EPS) { /* Converge */ break; }
        }

        // M-Step: Update model parameters
        UpdateBeta(beta, lambda, V, K);
        UpdateAlpha();
    }

    // TODO: Evaluate
    cout << lambda << endl;
    for(int k = 0; k < K; k++){
        uvec indx = sort_index(lambda.row(k).t(), "descend");
        cout << "topic " << k << endl;
        for(int i = 0; i < 5; i++){
            cout << idx2word[indx(i)] << ' ' << lambda(k, indx(i)) << endl;
        }
        cout << endl;
    }

    return 0;
}
