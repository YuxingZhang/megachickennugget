#include <fstream>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <set>
#include <algorithm>
using namespace std;
#include "load.cpp"

void load_files(string embedding, string corpus, double** word_embd, map<string, int>& word2idx, map<int, string>& idx2word, vector<vector<string> >& W, vector<int>& N);
double RandomDoubleVector(double v[]) {
    double sum = 0.0;
    for (size_t i = 0; i < (sizeof(v) / sizeof(v[0])); i++) {
        v[i] = ((double) rand() / (RAND_MAX));
        sum += v[i];
    }
    return sum;
}

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
    vector<double**> Z; // each element is a n_d * K matrix 
    for (int i = 0; i < D; i++) {
        double** tmp = new double*[N[i]];
        for (int j = 0; j < N[i]; j++) {
            tmp[j] = new double[K];
            double sum = RandomDoubleVector(tmp[j]);
            for (int k = 0; k < K; k++) {
                tmp[j][k] /= sum;
            }
        }
        Z.push_back(tmp);
    }

    double eta_m[D][K]; // mean of eta
    double eta_s[D][K]; // sigma of eta
    for (int i = 0; i < D; i++) {
        RandomDoubleVector(eta_m[i]);
        RandomDoubleVector(eta_s[i]);
    }

    double a_m[D][DOC_DIM];
    double a_s[DOC_DIM][DOC_DIM]; // all a_d share the same matrix
    for (int i = 0; i < D; i++) {
        RandomDoubleVector(a_m[i]);
    }
    for (int i = 0; i < DOC_DIM; i++) {
        a_s[i][i] = ((double) rand() / (RAND_MAX));
    }

    double rho_m[K][V]; // mean of rho
    double rho_s[K][V]; // sigma of rho
    for (int i = 0; i < K; i++) {
        RandomDoubleVector(rho_m[i]);
        RandomDoubleVector(rho_s[i]);
    }

    double up_m[K][WORD_DIM]; // mean of u prime
    double up_s[WORD_DIM][WORD_DIM]; // sigma of u prime, shared
    for (int i = 0; i < K; i++) {
        RandomDoubleVector(up_m[i]);
    }
    for (int i = 0; i < WORD_DIM; i++) {
        up_s[i][i] = ((double) rand() / (RAND_MAX));
    }

    double u_m[K][DOC_DIM]; // mean of u
    double u_s[DOC_DIM][DOC_DIM]; // sigma of u, shared
    for (int i = 0; i < K; i++) {
        RandomDoubleVector(u_m[i]);
    }
    for (int i = 0; i < DOC_DIM; i++) {
        u_s[i][i] = ((double) rand() / (RAND_MAX));
    }

    double xi_KW[K][V]; // used in the lower bound of z_dn and rho
    double alpha_K[K];
    for (int i = 0; i < K; i++) {
        RandomDoubleVector(xi_KW[i]);
    }
    RandomDoubleVector(alpha_K);

    double xi_DK[D][K]; // used in the lower bound of eta_d
    double alpha_D[D];
    for (int i = 0; i < D; i++) {
        RandomDoubleVector(xi_DK[i]);
    }
    RandomDoubleVector(alpha_D);

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
    /*while (true) {
        iteration++;

        while (true) {
            ComputeUpSigma(up_s, word_embedding, beta, l, WORD_DIM, V);

            bool has_converge = true;
            cur_batch--;
            if (cur_batch < 0) {
                cur_batch += num_of_batch;
            }

            // create a batch index set
            set<int> idx_set;
            for (int i = cur_batch * BATCH_SIZE; i < (cur_batch + 1) * BATCH_SIZE && i < random_index.size(); i++) {
                idx_set.insert(random_index[i]);
            }

            for (int d = idx_set.begin(); d != idx_set.end(); d++) {
                for (int n = 0; n < N[d]; n++) {
                    ;
                }
            }
        }
    }*/
    return 0;

}

