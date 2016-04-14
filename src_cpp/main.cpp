#include <fstream>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
//#include "load.cpp"

using namespace std;

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

    double** word_embedding; // V * dim, each line is a vector of double
    map<string, int> word2idx; // V
    map<int, string> idx2word; // V
    vector< vector<string> > W; // W[d][n] is w_dn
    int* N; // N[d] is the length of document d
    //load_files(emb_file, corpus_file, word_embedding, word2idx, idx2word, W, N);

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
        double tmp[N[i]][K];
        for (int j = 0; j < N[i]; j++) {
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

}

