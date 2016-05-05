import re
import string


if __name__ == '__main__':
	remove = set(['the', 'will', 'a', 'this', 'that', 'of', 'an', 'to', 'as', 'is', 'are', 'was', \
		'for', 'where', 'with', 'which', 'in', 'and', 'by', 'we', 'be', 'can', 'on', 'at', 'from',\
		'some', 'too', 'it', 'not', 'or', 'were', 'one', 'then', 'if', 'all', 'when', 'there', 'must',\
		'how', 'no', 'since', 'very', 'however', 'any', 'just', 'only', 'also', 're', 'subject'])

	path = '/Users/Lidan/Documents/CMU Yr1 Sem2/10-708/Project/newsdata/'
	input_file = 'a_result/corpus.txt'
	# new_corpus_file gives the processed corpus file without 'remove' words
	new_corpus_file = 'a_result/new_corpus.txt'
	output_file = 'a_result/wordcount.dat'
	output_dict_file = 'a_result/vocab.dat'

	vocab = list()
	vocab_hm = dict()

	f = open(path + input_file, 'r')
	f_out = open(path + output_file, 'w')
	f_out_dict = open(path + output_dict_file, 'w')

	# vocab keeps a record of the words appeared in the corpus
	# vocab_hm keep the number of occurance of each word in the corpus
	for l in f.readlines():
		l = l[:-1]
		words = l.split(' ')
		if len(words) < 200:
			for w in words:
				if (len(w) > 1) and (not w in remove):
					if not vocab_hm.has_key(w): 
						vocab.append(w)
						vocab_hm[w] = 1
					else:
						vocab_hm[w] += 1
	f.close()

	# remove words that appear fewer than 50 times and output the vocab.dat file
	for i in range(len(vocab)):
		w = vocab[i]
		if vocab_hm[w] < 50:
			vocab_hm.pop(w)

	# make a new list of vocab after removing the words
	print 'total vocabs ' + str(len(vocab))
	vocab = list()
	for w in vocab_hm:
		vocab.append(w)
		f_out_dict.write(w + '\n')
	print 'total vocabs ' + str(len(vocab))
	f_out_dict.close()

	f = open(path + input_file, 'r')
	fout_corpus = open(path + new_corpus_file, 'w')
	for l in f.readlines():
		d = dict()
		l = l[:-1]
		words = l.split(' ')
		# only keep records with fewer than 2000 words
		if len(words) < 200:
			for w in words:
				if vocab_hm.has_key(w):
					if len(d) > 0:
						fout_corpus.write(' ');
					fout_corpus.write(w)
					if d.has_key(w):
						d[w] += 1
					else:
						d[w] = 1
			f_out.write(str(len(d)))
			for i in range(len(vocab)):
				if d.has_key(vocab[i]):
					f_out.write(' ' + str(i) + ':' + str(d[vocab[i]]))
			if len(d) > 0:
				f_out.write('\n')
				fout_corpus.write('\n')
	f.close()
	f_out.close()
	fout_corpus.close()
