import re
import string


if __name__ == '__main__':

    path = '/home/johnny/Desktop/Probabilistic Graphical Models/10708 project/nipsdata/'
    input_file = 'file_names'
    output_file = 'corpus.txt'
    
    f = open(path + input_file, 'r')
    docs = f.readlines()
    f.close()
    fout = open(path + output_file, 'w')
    
    for line in docs:
        f = open(path + line.strip(), 'r')
        doc = f.readlines()
        f.close()
        for line in doc:
            line = re.sub(r"\d+", '', line)
            line = re.sub(r"[^\w\s]", '', line)
            line = re.sub(r"\s+", ' ', line)
            if line.strip() != '':
                fout.write(line.strip() + ' ')

        fout.write('\n')

    fout.close()
