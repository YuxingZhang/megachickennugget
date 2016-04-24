/*
 * Used to test the correctness of each function.
 */

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
#include "load.cpp"
#include "update.cpp"

void load_files(string embedding, string corpus, mat word_embd, map<string, int>& word2idx, map<int, string>& idx2word, vector<vector<string> >& W, vector<int>& N);

int main() {
    // Reading input files, including the corpus and the embedding
    string emb_file = "../vectors_2.txt";
    string corpus_file = "../new_corpus_2.txt";

    map<string, int> word2idx;  // V
    map<int, string> idx2word;  // V
    vector< vector<string> > W; // W[d][n] is w_dn
    vector<int> N; // N[d] is the length of document d
    mat word_embedding = load_files(emb_file, corpus_file, word2idx, idx2word, W, N); // V * dim, each line is a vector of double

    // some parameters
    const int V = idx2word.size(); // vocabulary size
    const int D = W.size(); // number of documents
    const int K = 2; // number of topics
    const int WORD_DIM = 2; // dimension of word embedding
    const int DOC_DIM = 2;// dimension of document embedding

    // model parameters, changed in the M step
    double l = 1.0; 
    double c = 1.0;
    double kappa = 1.0;
    double beta = 1.0;
    double gamma = 1.0;

    // initialization
    vector<mat> z; // each element is a n_d * K matrix 
    for (int i = 0; i < D; i++) {
        mat tmp(N[i], K, fill::ones);
        tmp = normalise(tmp, 1, 1);
        cout << "======= z_" << i << "=======" << endl;
        cout << tmp << endl;
        z.push_back(tmp);
    }
    cout << "========================" << endl;

    mat eta_m(D, K, fill::ones); // mean of eta
    mat eta_s(D, K, fill::ones); // sigma of eta

    mat a_m(D, DOC_DIM, fill::ones);
    mat a_s = diagmat(vec(DOC_DIM, fill::ones)); // all a_d share the same matrix

    mat rho_m(K, V, fill::ones); // mean of rho
    mat rho_s(K, V, fill::ones); // sigma of rho

    mat up_m(K, WORD_DIM, fill::ones); // mean of u prime
    mat up_s = diagmat(vec(WORD_DIM, fill::ones)); // sigma of u prime, shared

    mat u_m(K, DOC_DIM, fill::ones); // mean of u
    mat u_s = diagmat(vec(DOC_DIM, fill::ones)); // sigma of u, shared

    mat xi_KW(K, V, fill::ones); // used in lower bound of z_dn and rho
    vec alpha_K(K, fill::ones);

    mat xi_DK(D, K, fill::ones); // used in lower bound of eta_d
    vec alpha_D(D, fill::ones);

    // train for each batch
    vector<int> random_index;
    for (size_t i = 0; i < W.size(); i++) {
        random_index.push_back(i);
    }
    random_shuffle(random_index.begin(), random_index.end());
    const int BATCH_SIZE = 2;
    const double EPS = 0.1;
    int num_of_batch = (int)((random_index.size() + BATCH_SIZE - 1) / BATCH_SIZE);
    int cur_batch = num_of_batch;
    cout << lambda(1.0) << endl;
    cout << "====================" << endl;

    //ComputeUpSigma(up_s, word_embedding, beta, l, WORD_DIM, V);
    //UpdateUp(1, up_m, up_s, rho_m, word_embedding, beta, WORD_DIM, V, EPS);
    //UpdateEta(0, eta_m, eta_s, xi_DK, alpha_D, u_m, a_m, z, gamma, N, K, EPS);
    //UpdateZ(0, 0, z, eta_m, rho_m, rho_s, xi_KW, alpha_K, W, word2idx, K, V, EPS);
    for (int d = 0; d < 2; d++) {
        UpdateAuxiliary(d, alpha_D, xi_DK, eta_m, eta_s, K);
    }
    cout << alpha_D << endl;
    cout << "====================" << endl;
    cout << xi_DK << endl;
    return 0;

    int iteration = 0;
    int MAX_ITER = 10;
    while (iteration < MAX_ITER) {
        iteration++;
        ComputeUpSigma(up_s, word_embedding, beta, l, WORD_DIM, V);

        int inner_iteration = 0;

        while (true) {
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

            for (set<int>::iterator d = idx_set.begin(); d != idx_set.end(); d++) {
                /*
                for (int n = 0; n < N[*d]; n++) {
                    cout << "============================= z_dn "  << *d << "   " << n<< "=============================" << endl;
                    cout << z[*d].row(n) << endl;
                }
                cout << "----------------------------- after -------------------------------" << endl;
                */
                UpdateAuxiliary(*d, alpha_D, xi_DK, eta_m, eta_s, K);
                for (int n = 0; n < N[*d]; n++) {
                    if (!UpdateZ(*d, n, z, eta_m, rho_m, rho_s, xi_KW, alpha_K, W, word2idx, K, V, EPS)) { has_converge = false; }
                    //cout << "============================= z_dn "  << *d << "   " << n<< "=============================" << endl;
                    //cout << z[*d].row(n) << endl;
                }
                if (!UpdateEta(*d, eta_m, eta_s, xi_DK, alpha_D, u_m, a_m, z, gamma, N, K, EPS)) { has_converge = false; }
                if (!UpdateA(*d, a_m, a_s, u_m, u_s, eta_m, c, gamma, DOC_DIM, K, EPS)) { has_converge = false; }
            }

            for (int k = 0; k < K; k++) {
                UpdateAuxiliary(k, alpha_K, xi_KW, rho_m, rho_s, V);
                if (!UpdateRho(k, rho_m, rho_s, z, up_m, alpha_K, xi_KW, word_embedding, W, idx2word, beta, D, N, V, EPS)) { has_converge = false; }
                //cout << "============================= rho_m "  << k << "=============================" << endl;
                //cout << rho_m.row(k) << endl;
                if (!UpdateU(k, u_m, u_s, a_m, a_s, eta_m, kappa, gamma, DOC_DIM, D, EPS)) { has_converge = false; }
                if (!UpdateUp(k, up_m, up_s, rho_m, word_embedding, beta, WORD_DIM, V, EPS)) { has_converge = false; }
            }
            if (has_converge) { break; }
            return 0;
        }

        l = UpdateL(up_m, up_s, WORD_DIM, K);
        kappa = UpdateKappa(u_m, u_s, DOC_DIM, K);
        c = UpdateC(a_m, a_s, DOC_DIM, D);
        beta = UpdateBeta(up_m, up_s, word_embedding, rho_m, rho_s, V, K);
        gamma = UpdateGamma(eta_m, eta_s, a_m, a_s, u_m, u_s, D, K);
    }

    // TODO: Evaluate
    mat phi = exp(rho_m);
    for(int k = 0; k < K; k++){
        phi.row(k) = normalise(phi.row(k), 1);
        cout << phi.row(k) << endl;
	cout << rho_m.row(k) << endl;
        uvec indx = sort_index(phi.row(k).t(), "descend");
        cout << "topic " << k << endl;
        for(int i = 0; i < 5; i++){
            cout << idx2word[indx(i)] << endl;
        }
        cout << endl;
    }

    return 0;
}
