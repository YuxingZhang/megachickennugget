import re
import string


if __name__ == '__main__':
	path = '/Users/Lidan/Documents/CMU Yr1 Sem2/10-708/Project/nipsdata/'
	input_file = 'a_result/corpus.txt'
	output_file = "a_result/abstracts.txt"

	fin = open(path + input_file, 'r')
	fout = open(path + output_file, 'w')
	
	for line in fin.readlines():
		line = line[:-1]
		words = line.split(' ')
		start = -1
		end = -1
		for i in range(len(words)):
			if start != -1 and end != -1:
				break
			if start == -1 and words[i] == "abstract":
				start = i
			if end == -1 and words[i] == "introduction":
				end = i
		start += 1
		for i in range(start, end):
			if i != start:
				fout.write(' ')
			fout.write(words[i])
		fout.write('\n')
	fin.close()
