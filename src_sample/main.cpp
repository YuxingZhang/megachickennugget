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
    string emb_file = "../mini_abs_vectors.txt";
    string corpus_file = "../mini_abs_corpus.txt";
    //arma_rng::set_seed_random();
    map<string, int> word2idx;  // V
    map<int, string> idx2word;  // V
    vector< vector<string> > W; // W[d][n] is w_dn
    vector<int> N; // N[d] is the length of document d
    load_files(emb_file, corpus_file, word2idx, idx2word, W, N); // V * dim, each line is a vector of double

    // some parameters
    const int V = idx2word.size(); // vocabulary size
    const int D = W.size(); // number of documents
    const int K = 10; // number of topics
    const int WORD_DIM = 20; // dimension of word embedding
    const int DOC_DIM = 3;// dimension of document embedding

    // model parameters, changed in the M step
    double c = 1;
    double kappa = 1;
    double gamma = 0.1;
    vec beta(V, fill::zeros);
    beta += 1;

    // initialization
    vector<mat> z; // each element is a n_d * K matrix 
    for (int i = 0; i < D; i++) {
        mat tmp(N[i], K, fill::randu);
        tmp = normalise(tmp, 1, 1);
        z.push_back(tmp);
    }

    mat eta_m(D, K, fill::randu); // mean of eta
    mat eta_s(D, K, fill::randu); // sigma of eta

    mat a_m(D, DOC_DIM, fill::randu);
    mat a_s = diagmat(vec(DOC_DIM, fill::randu)); // all a_d share the same matrix

    mat rho_m(K, V, fill::randu); // mean of rho

    mat u_m(K, DOC_DIM, fill::randu); // mean of u
    mat u_s = diagmat(vec(DOC_DIM, fill::randu)); // sigma of u, shared

    // train for each batch
    vector<int> random_index;
    for (size_t i = 0; i < W.size(); i++) {
        random_index.push_back(i);
    }
    random_shuffle(random_index.begin(), random_index.end());
    const int BATCH_SIZE = 1000;
    const double EPS = 0.1;
    int num_of_batch = (int)((random_index.size() + BATCH_SIZE - 1) / BATCH_SIZE);
    int cur_batch = num_of_batch;

    cout << "vocabulary size = " << V << endl;

    int iteration = 0;
    int MAX_ITER = 10;
    int aux_iter = 10;
    while (iteration < MAX_ITER) {
        iteration++;

        int inner_iteration = 0;
        while (inner_iteration < 2 * MAX_ITER) {
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

            if (!UpdateZ(idx_set, N, z, eta_m, rho_m, W, word2idx, K, EPS)) { 
                has_converge = false; 
            }
            for (set<int>::iterator d = idx_set.begin(); d != idx_set.end(); d++) {
                SampleEta(*d, eta_m, u_m, a_m, z, gamma, N, K);
                if (!UpdateA(*d, a_m, a_s, u_m, u_s, eta_m, c, gamma, DOC_DIM, K, EPS)) {
                    has_converge = false;
                }
            }

            for (int k = 0; k < K; k++) {
                if (!UpdateRho(k, rho_m, z, W, word2idx, beta, D, N, V, EPS)) {
                    has_converge = false; 
                }
                if (!UpdateU(k, u_m, u_s, a_m, a_s, eta_m, kappa, gamma, DOC_DIM, D, EPS)) {
                    has_converge = false; 
                }
            }
            cout << "iteration finished" << endl;
            if (has_converge) { break; }
        }

        //kappa = UpdateKappa(u_m, u_s, DOC_DIM, K);
        //c = UpdateC(a_m, a_s, DOC_DIM, D);
    }

    ofstream fout;
    fout.open ("u_m.txt");
    fout << u_m << endl;
    fout.close();
	
    fout.open ("a_m.txt");
    fout << a_m << endl;
    fout.close();
    
    fout.open ("eta_m.txt");
    fout << eta_m << endl;
    fout.close();

    fout.open ("rho_m.txt");
    fout << rho_m << endl;
    fout.close();
        
    fout.open("topic.txt");
    // TODO: Evaluate
    for(int k = 0; k < K; k++){
        //cout << rho_m.row(k) << endl;
        uvec indx = sort_index(rho_m.row(k).t(), "descend");
        fout << "topic " << k << endl;
        for(int i = 0; i < 20; i++){
            fout << idx2word[indx(i)] << endl;
        }
        fout << endl;
    }
    fout.close();
    return 0;
}
