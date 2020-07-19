import os

SRC = "inputs/parlAIopeninglines.txt"
FILENAME = "train_none_original_no_cands.txt"

def parse_data(filename):
    src = open(SRC, 'w')
    inputstr = ""
    outputstr = ""
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().split(" ")
            linenum = int(line[0])
            line = ' '.join(line[1:])
            if line.startswith("__SILENCE__"):
                print("silence")
                break
            line = line.split('\t')
            if linenum == 1:
                src.write(line[0] + "\n")

    
    src.close()




if __name__ == "__main__":
    parse_data(FILENAME)