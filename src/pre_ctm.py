import re
import string


if __name__ == '__main__':
	remove = set(['the', 'will', 'a', 'this', 'that', 'of', 'an', 'to', 'as', 'is', 'are', 'was', \
		'for', 'where', 'with', 'which', 'in', 'and', 'by', 'we', 'be', 'can', 'on', 'at', 'from',\
		'some', 'too', 'it', 'not', 'or', 'were', 'one', 'then', 'if', 'all', 'when', 'there', 'must',\
		'how', 'no', 'since', 'very', 'however', 'any', 'just', 'only', 'also'])

	path = '/Users/Lidan/Documents/CMU Yr1 Sem2/10-708/Project/newsdata/'
	input_file = 'new_corpus.txt'
	output_file = 'new_wordcount.dat'
	output_dict_file = 'new_vocab.dat'

	vocab = list()
	vocab_hm = dict()

	f = open(path + input_file, 'r')
	f_out = open(path + output_file, 'w')
	f_out_dict = open(path + output_dict_file, 'w')

	for l in f.readlines():
		l = l[:-1]
		words = l.split(' ')
		for w in words:
			if (len(w) > 1) and (not w in remove):
				if not vocab_hm.has_key(w): 
					vocab.append(w)
					vocab_hm[w] = 1
					f_out_dict.write(w + '\n')
				else:
					vocab_hm[w] += 1
	f.close()
	f_out_dict.close()

	print 'total vocabs' + str(len(vocab))
	for i in range(len(vocab)):
		print vocab[i], ' ', vocab_hm[vocab[i]]

	f = open(path + input_file, 'r')
	for l in f.readlines():
		d = dict()
		l = l[:-1]
		words = l.split(' ')
		for w in words:
			if vocab_hm.has_key(w):
				if d.has_key(w):
					d[w] += 1
				else:
					d[w] = 1
		f_out.write(str(len(d)))
		for i in range(len(vocab)):
			if d.has_key(vocab[i]):
				f_out.write(' ' + str(i) + ':' + str(d[vocab[i]]))
		f_out.write('\n\n')
	f.close()
	f_out.close()
