#include <fstream>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
//#include "load.cpp"

using namespace std;

int main() {
    // Reading input files, including the corpus and the embedding
    string emb_file = "../vectors.txt";
    string corpus_file = "../new_corpus.txt";

    double** word_embedding;
    map<string, int> word2idx;
    string* idx2word;
    vector<vector<string>> W;
    int* N;
    load_files(emb_file, corpus_file, word_embedding, word2idx, idx2word, W, N);
}
