import re
import string

filein = open(input_file, 'r')
fileout = open(output_file)
pattern = re.compile('[\W_]+')
for line in file:
	for word in line.strip().split():
		new_word = pattern.sub('', word),