import argparse
import time
import torch
from Models import get_model
from Process import *
import torch.nn.functional as F
from Batch import create_masks
import pdb
import dill as pickle
import argparse
from Models import get_model
from Beam import beam_search
from nltk.corpus import wordnet
from torch.autograd import Variable
import re
import pyttsx3

def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]
            
    return 0

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  value = regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 
  return value

def generate_sentence(sentence, model, opt, SRC, TRG):
    
    model.eval()
    indexed = []
    sentence = SRC.preprocess(sentence)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = Variable(torch.LongTensor([indexed]))
    if opt.device == 0:
        sentence = sentence.cuda()
    
    sentence = beam_search(sentence, model, SRC, TRG, opt)

    return  multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, sentence)

def getSentence(opt, model, SRC, TRG):
    sentences = opt.text.lower().strip().split('.')
    sentences = [opt.text.lower().strip()]
    sent = []

    for sentence in sentences:
        sentence = sentence.strip()
        if sentence[-1].isalpha():
            sentence+="."
        sent.append(generate_sentence(sentence, model, opt, SRC, TRG).capitalize())

    return (' '.join(sent))


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights')
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')

    
    opt = parser.parse_args()
    

    opt.device = 0 if opt.no_cuda is False else -1
 
    assert opt.k > 0
    assert opt.max_len > 10


    opt.src_lang = "en_core_web_sm"
    opt.trg_lang = "en_core_web_sm"
    opt.load_weights = "weights"
    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    
    sentence = []

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 150)


    #main loop to talk to bot
    while True:
        userinput = input(">> ")
        userinput = userinput.strip()
        if userinput[-1].isalpha():
            userinput+= "."
        if len(sentence) >= 5:
            sentence.pop(0)
        sentence.append(userinput)
    
        opt.text = ' '.join(sentence)
        
        phrase = getSentence(opt, model, SRC, TRG)
        print('chatbot: '+ phrase)

        #makes chat bot speak out loud
        engine.say(phrase)
        engine.runAndWait()

if __name__ == '__main__':
    main()
