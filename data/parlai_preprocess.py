import os

SRC = "inputs/parlAIinput.txt"
TRG = "inputs/parlAIoutputs.txt"
FILENAME = "train_none_original_no_cands.txt"

def parse_data(filename):
    src = open(SRC, 'w')
    trg = open(TRG, 'w')
    inputstr = ""
    outputstr = ""
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().split(" ")
            linenum = int(line[0])
            if linenum == 1:
                inputstr = ""
                outputstr = ""

            line = ' '.join(line[1:])
            if line.startswith("__SILENCE__"):
                break
            line = line.strip("__SILENCE__").strip()
            line = line.split('\t')

            if len(line) != 2:
                print("line: " + str(line))
                print("error. invalid pair. length: " + str(len(line)))
                src.close()
                trg.close()
                raise Exception

            inputstr += line[0]
            outputstr += line[1]
            src.write(inputstr+"\n")
            trg.write(line[1] + "\n")
    src.close()
    trg.close()





if __name__ == "__main__":
    parse_data(FILENAME)