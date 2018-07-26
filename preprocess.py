from multiprocessing import Pool
import re
import string
translator=str.maketrans('','',string.punctuation)

def processline(line):
    line = line.strip().lower()
    line=line.translate(translator)
    line = re.sub('“|”', ' ', line)

    spaceline = line
    line = re.sub(' ', '', line)
    return line +','+ spaceline

if __name__ == '__main__':
    fin = open('data/VNESEcorpus.txt')
    lines = fin.readlines()
    p = Pool(5)
    lines = p.map(processline, lines)
    fout = open('data/corpus', 'w')
    fout.write('\n'.join(lines))
