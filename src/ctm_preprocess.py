import re
import string


if __name__ == '__main__':

	path = '/Users/tianshuren/Google Drive/2016 Spring/10708/megachickennugget/src/'
	input_file = 'corpus.txt'
	output_file = 'wordcount.txt'
	output_dict_file = 'vocab.dat'

	vocab = list()
	vocab_hm = dict()

	f = open(path + input_file, 'r')
	f_out = open(path + output_file, 'w')
	f_out_dict = open(path + output_dict_file, 'w')

	i = 0
	for l in f.readlines():
		l = l[:-1]
		words = l.split(' ')
		for w in words:
			if not vocab_hm.has_key(w):
				vocab.append(w)
				vocab_hm[w] = 1
				f_out_dict.write(w + '\n')
				i += 1
	f.close()
	f_out_dict.close()

	print 'total vocabs' + str(len(vocab))

	f = open(path + input_file, 'r')
	for l in f.readlines():
		d = dict()
		l = l[:-1]
		words = l.split(' ')
		for w in words:
			if d.has_key(w):
				d[w] += 1
			else:
				d[w] = 1
		f_out.write(str(len(d)))
		i = 0
		for i in range(len(vocab)):
			if d.has_key(vocab[i]):
				f_out.write(' ' + str(i) + ':' + str(d[vocab[i]]))
			# else:
				# f_out.write(' ' + str(i) + ':' + str(0))
			# f_out_dict.write(str(i) + ':' + k + ' ')
			# i += 1
		f_out.write('\n')
		# f_out_dict.write('\n')
	f.close()
	f_out.close()


	#
    # f = open(path + input_file, 'r')
    # docs = f.readlines()
    # f.close()
    # fout = open(path + output_file, 'w')
    #
    # for line in docs:
    #     f = open(path + line.strip(), 'r')
    #     doc = f.readlines()
    #     f.close()
    #     for w in doc:
    #         line = re.sub(r"\d+", '', line)
    #         line = re.sub(r"[^\w\s]", '', line)
    #         line = re.sub(r"\s+", ' ', line)
    #         if line.strip() != '':
    #             fout.write(line.strip() + ' ')
	#
    #     fout.write('\n')
	#
    # fout.close()
