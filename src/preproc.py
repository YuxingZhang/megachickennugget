import re
import string


if __name__ == '__main__':

    path = '/Users/Lidan/Documents/CMU Yr1 Sem2/10-708/Project/newsdata/'
    input_file = 'file_names'
    mid_file = 'output.txt'
    output_file = 'corpus.txt'

    
    f = open(path + input_file, 'r')
    docs = f.readlines()
    f.close()
    fmid = open(path + mid_file, 'w')
    for files in docs:
        f = open(path + files.strip(), 'r')
        for line in f:
            fmid.write(line.strip() + ' ')
        f.close()
        fmid.write('\n')
    fmid.close()

    f = open(path + mid_file, 'r')
    fout = open(path + output_file, 'w')
    for line in f:
        line = re.sub("- ", '', line)
        line = re.sub("-", " ", line)
        line = re.sub("_", '', line)
        line = re.sub(r"\d+", '', line)
        line = re.sub(r"[^\w\s]", '', line)
        line = re.sub(r"\s+", ' ', line)
        line = line.lower()
        fout.write(line.strip())
        fout.write('\n')
    fout.close()
    f.close()
