import re
import string
translator=str.maketrans('','',string.punctuation)

def processline(line):
    line = line.lower()
    line=line.translate(translator)
    line = re.sub(' ', '', line)
    return line

if __name__ == '__main__':
    fin = open('data/VNESEcorpus.txt')
    lines = fin.readlines()
    print(processline(lines[0]))
