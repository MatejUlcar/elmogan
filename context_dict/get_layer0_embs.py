from allennlp.commands.elmo import ElmoEmbedder
import numpy as np
#from scipy import spatial
#import time
#from math import floor
#import heapq
import argparse
import random
#from annoy import AnnoyIndex

#options_file = "path/to/elmo_model/options.json"
#weight_file = "path/to/elmo_model/gigafida_weights.hdf5"
#options_file = "../elmo_model/29_apr/options.json"
#weight_file = "../elmo_model/gigafida_weights.hdf5"
parser = argparse.ArgumentParser()
parser.add_argument('-w1', '--weights1', required=True, help="Path to elmo weights file (.hdf5)")
parser.add_argument('-o1', '--options1', required=True, help="Path to elmo options file (.json)")
parser.add_argument('-w2', '--weights2', required=True, help="Path to elmo weights file (.hdf5)")
parser.add_argument('-o2', '--options2', required=True, help="Path to elmo options file (.json)")
parser.add_argument('-d', '--dictionary', required=True, help="Path to dictionary file (one pair per line).")
#parser.add_argument('--side', choices=['left', 'right'], help="Pick left or right side of dictionary.")
parser.add_argument('-e', '--output', required=True, help="Output folder of embeddings.")
parser.add_argument('-l1', '--language1')
parser.add_argument('-l2', '--language2')
args = parser.parse_args()


#csd = spatial.distance.cosine
def write_embs(output, embs):
    with open(output, 'w') as outfile:
        buffer = str(len(embs))+" 1024\n"
        wordcount = 0
        for word in embs:
            buffer += word+' '+' '.join([str(v) for v in embs[word]])+'\n'
            wordcount += 1
            if wordcount >= 1000:
                wordcount = 0
                outfile.write(buffer)
                buffer = ""
        outfile.write(buffer)


#vocab = []
embs = {}
def load_dictionary(args):
    with open(args.dictionary, 'r') as tsvdict:
        dictionary = {} 
        for line in tsvdict:
            line = line.strip().split('\t')
            if ' ' not in line[0] and ' ' not in line[1]:
                if line[0] in dictionary:
                    dictionary[line[0]].append(line[1])
                else:
                    dictionary[line[0]] = [line[1]]
    return dictionary


def calc_embs(vocab, elmo):
    embs = {}
    embcounter = 0
    minibatches = [vocab[i:i+10] for i in range(0, len(vocab), 10)]
    for i in range(0, len(minibatches), 10):
        batchtokens = minibatches[i:i+10]
        if (len(batchtokens), len(batchtokens[0]), len(batchtokens[-1])) != (10,10,10):
            print(len(batchtokens), len(batchtokens[0]), len(batchtokens[-1]))
        batchvectors = elmo.embed_batch(batchtokens)
        if (len(batchvectors), len(batchvectors[0][0]), len(batchvectors[-1][0])) != (10,10,10):
            print(len(batchvectors), len(batchvectors[0][0]), len(batchvectors[-1][0]))
        for s in range(len(batchvectors)):
            for w in range(len(batchvectors[s][0])):
                embs[batchtokens[s][w]] = batchvectors[s][0][w]
                embcounter += 1
                #if embcounter%5000 == 0:
                #    print(embcounter, '/', len(vocab))
    print(len(embs))
    print(embcounter)
    return embs

dictionary = load_dictionary(args)
vocab1 = list(dictionary.keys())
vocab2 = [w for words in dictionary.values() for w in words]
elmo = ElmoEmbedder(args.options1, args.weights1, 0)
embs1 = calc_embs(vocab1, elmo)
elmo = ElmoEmbedder(args.options2, args.weights2, 0)
embs2 = calc_embs(vocab2, elmo)
write_embs(args.output+'/'+args.language1+'-'+args.language2+'_'+args.language1+'-layer00.emb', embs1)
write_embs(args.output+'/'+args.language1+'-'+args.language2+'_'+args.language2+'-layer00.emb', embs2)

with open(args.output+'/'+args.language1+'-'+args.language2+'.layer0.dict.train', 'w') as traindict, open(args.output+'/'+args.language1+'-'+args.language2+'.layer0.dict.eval', 'w') as evaldict:
    for entry in dictionary:
        for v in dictionary[entry]:
            r = random.random()
            if r <= 0.015:
                evaldict.write(entry+" "+v+'\n')
            else:
                traindict.write(entry+" "+v+'\n')
