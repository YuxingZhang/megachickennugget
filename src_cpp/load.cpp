mat load_files(string embedding, string corpus, map<string, int>& word2idx, map<int, string>& idx2word, vector<vector<string> >& W, vector<int>& N) {
    /* param:
     * embedding: file containing the word embedding results
     * corpus: file containing the original corpus
     * word_embd: store the word embedding vectors
     * word2idx: word in the vocabulary to its index in the word_embd vector
     * indx2word: index in the word_embd vector to word in the vocabulary
     * W: store the words in the corpus
     * N: store the length of each document
     * */
    ifstream embd_file;
    int vocabulary_size, embedding_dim, i, index, length;
    string val;
    char end;

    /* read in the word embedding file */
    embd_file.open(embedding.c_str());
    if(embd_file.is_open()){
    	/* record the vocabulary size and embedding dimension */
    	embd_file >> val;
	    vocabulary_size = stoi(val);
	    embd_file >> val;
	    embedding_dim = stoi(val);

	    
	    mat word_embd(vocabulary_size, embedding_dim);
	    index = 0;
	    /* store the word embedding results */
	    while(embd_file >> val){
			word2idx[val] = index;
			idx2word[index] = val;
	    	for(i = 0; i < embedding_dim; i++){
	    		embd_file >> val;
	    		word_embd(index, i) = stod(val);
	    	}
	    	index++;
	    }
	    embd_file.close();
	}

	ifstream corpus_file;
	corpus_file.open(corpus.c_str());
	
	/* read in the corpus file */
	if(corpus_file.is_open()){
		vector<string> line;
		length = 0;
		while(corpus_file >> val){
			length++;
			line.push_back(val);
			corpus_file.get(end);
			if(end == '\n'){
				N.push_back(length);
				W.push_back(line);
				length = 0;
				line.clear();
			}
		}
	}
    return word_embd;
}