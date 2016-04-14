#include <fstream>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include "load.cpp"

using namespace std;

int main() {
    // Reading input files, including the corpus and the embedding
    const char* str = "../vectors.txt";
    ifstream emb_in(str);

    double embedding[10]; // TODO change the size
}
