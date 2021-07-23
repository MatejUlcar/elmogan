import stanza
from allennlp.commands.elmo import ElmoEmbedder
import argparse
import lzma
import os
import re
# IMPORT LEMMATIZER
# LOAD EMBEDDERS
# LOAD DICT AS PYDICT
# LOAD PARALLEL CORPUS
## FOR EACH LINE/PARAGRAPH DO
### EMBED BOTH LANGS (each with diff embedder)
### LEMMATIZE BOTH LANGS (nlp(line))
### FOR EACH WORD/LEMMA IN PAR_LANG1:
#### IF DICT[LEMMA] IN PAR_LANG2:
##### ADD "uid_word1 uid_word2" TO NEW DICT
##### ADD "uid_word1 embedding(word1)" TO EMB_LANG1
##### ADD "uid_word2 embedding(word2)" TO EMB_LANG2
### IF LEN(NEW_DICT) >= 200000 THEN BREAK


    
def flatten_emb(emb):
    emb2 = []
    for l in range(3):
        emb2.append([word for sentence in emb for word in sentence[l]])
    return emb2
    
def add2condict(emb1, emb2, layer, word1, word2, num):
    if layer == 0:
        file_dict.write((word1+str(num)+' '+word2+str(num)+'\n').encode())
    lang1[layer].write((word1+str(num)+' '+' '.join([str(i) for i in emb1])+'\n').encode())
    lang2[layer].write((word2+str(num)+' '+' '.join([str(i) for i in emb2])+'\n').encode())
    
def batch_condict_write(entbuffer, args):
    l1 = ['','','']
    l2 = ['','','']
    d = ''
    for (emb1, emb2, layer, word1, word2, num) in entbuffer:
        if layer == 0:
            d += word1+str(num)+' '+word2+str(num)+'\n'
        l1[layer] += word1+str(num)+' '+' '.join([str(i) for i in emb1])+'\n'
        l2[layer] += word2+str(num)+' '+' '.join([str(i) for i in emb2])+'\n'
    if args.nocompress:
        file_dict.write(d)
    else:
        file_dict.write(d.encode())
    for layer in [0,1,2]:
        if args.nocompress:
            lang1[layer].write(l1[layer])
            lang2[layer].write(l2[layer])
        else:
            lang1[layer].write(l1[layer].encode())
            lang2[layer].write(l2[layer].encode())

def load_dictionary(args):
    with open(args.dictionary, 'r') as tsvdict:
        dictionary = {} 
        for line in tsvdict:
            line = line.strip().split('\t')
            if line[0] in dictionary:
                dictionary[line[0]].append(line[1])
            else:
                dictionary[line[0]] = [line[1]]
    return dictionary


def match_pars(ali1, ali2, args):
    # matches paragraphs from file ali1 with those from ali2, based on alignfile and outputs them
    # as generator, tab separated. sentences within paragraphs are </s> separated.

    with open(ali1, 'r') as leftpars, open(ali2, 'r') as rightpars:
        rline = False
        for line in leftpars:
            lpar = line.strip()
            rpar = rightpars.readline()
            rpar = rpar.strip()
            yield lpar, rpar

    
       
parser = argparse.ArgumentParser(description='Find words from languages L1 and L2 in same context, output dictionaries and embeddings for those words.')
parser.add_argument('-i', '--input1', help='L1 corpus in .vert format (or already pre-processed .ali format)')
parser.add_argument('-j', '--input2', help='L2 corpus in .vert format (or already pre-processed .ali format)')
parser.add_argument('-o', '--output', help='path to output folder')
parser.add_argument('-d', '--dictionary', help='L1-L2 dictionary')
parser.add_argument('-l1', '--lang1', help='L1 2-letter language code')
parser.add_argument('-l2', '--lang2', help='L2 2-letter language code')
parser.add_argument('--options1', help="Path to L1 elmo options file (.json)")
parser.add_argument('--options2', help="Path to L2 elmo options file (.json)")
parser.add_argument('--weights1', help="Path to L1 elmo weights file (.hdf5)")
parser.add_argument('--weights2', help="Path to L2 elmo weights file (.hdf5)")
parser.add_argument('--cuda', type=int, default=-1, help="which gpu to use with ELMo, set -1 for cpu")
parser.add_argument('--limit', type=int, default=20, help='max contexts per dictionary entry')
parser.add_argument('--hardmax', type=int, default=3000000, help='hard limit on max number of entries output')
parser.add_argument('--nocompress', action='store_true', help='do not xz compress the output')
parser.add_argument('--nonlemmatized', action='store_true', help='do not match word lemmas, but only actual word forms')
parser.add_argument('--splitpars', action='store_true', help='WIP /split paragraphs into individual sentences (will look for match only within sentence, not whole paragraph)')
parser.add_argument('--resume', help='resume from given contextual .dict')
args = parser.parse_args()



# LOAD LEMMATIZERS AND EMBEDDERS
os.environ["CUDA_VISIBLE_DEVICES"]="0"
if args.nonlemmatized:
    processors = 'tokenize'
else:
    processors = 'tokenize,pos,lemma'
try:
    nlp1 = stanza.Pipeline(lang=args.lang1, processors=processors)
except:
    stanza.download(lang=args.lang1, processors=processors)
    nlp1 = stanza.Pipeline(lang=args.lang1, processors=processors)
try:
    nlp2 = stanza.Pipeline(lang=args.lang2, processors=processors)
except:
    stanza.download(lang=args.lang2, processors=processors)
    nlp2 = stanza.Pipeline(lang=args.lang2, processors=processors)

print("Loading ELMo...")
elmo1 = ElmoEmbedder(args.options1, args.weights1, args.cuda)
elmo2 = ElmoEmbedder(args.options2, args.weights2, args.cuda)

# READ DICTIONARY
print("Loading dictionary...")
dictionary = load_dictionary(args)


# PREPARE OUTPUT FILES
if args.resume:
    write_flag = 'a'
else:
    write_flag = 'w'
if args.nocompress:
    file_dict = open(args.output+'/'+args.lang1+'-'+args.lang2+'.dict', write_flag)
else:
    file_dict = lzma.LZMAFile(args.output+'/'+args.lang1+'-'+args.lang2+'.dict.xz', write_flag)
lang1 = []
lang2 = []
for layer in ['0','1','2']:
    if args.nocompress:
        lang1.append(open(args.output+'/'+args.lang1+'-'+args.lang2+'_'+args.lang1+'-layer'+layer+'.emb', write_flag))
        lang2.append(open(args.output+'/'+args.lang1+'-'+args.lang2+'_'+args.lang2+'-layer'+layer+'.emb', write_flag))
    else: 
        lang1.append(lzma.LZMAFile(args.output+'/'+args.lang1+'-'+args.lang2+'_'+args.lang1+'-layer'+layer+'.emb.xz', write_flag))
        lang2.append(lzma.LZMAFile(args.output+'/'+args.lang1+'-'+args.lang2+'_'+args.lang2+'-layer'+layer+'.emb.xz', write_flag))
for layer in range(3):
    header_emb = str(args.hardmax)+' 1024\n'
    if args.nocompress:
        lang1[layer].write(header_emb)
        lang2[layer].write(header_emb)
    else:
        lang1[layer].write(header_emb.encode())
        lang2[layer].write(header_emb.encode())
# MATCH PARAGRAPHS    
matchingpars = match_pars(args.input1, args.input2, args)

# FIND WORD PAIRS AND THEIR EMBEDDINGS
appearances = {}
writebuffer = []
pars_seen = 0
num_entries = 0
tok1 = []
tok2 = []
lem1 = []
lem2 = []
already_processed = 0
if args.resume:
    with open(args.resume, 'r') as reader:
        for line in reader:
            already_processed += 1

for lpar, rpar in matchingpars:
    pars_seen += 1
    #if pars_seen > 2:
    #    break
    if num_entries >= args.hardmax:
        print('Max limit reached. Quitting.')
        break
    if pars_seen % 500 == 0:
        print('Seen/read',pars_seen,'paragraphs.')
        print(num_entries, 'entries written so far.')

    expand_lang = {'en': 'english', 'sl': 'slovene', 'et': 'estonian', 'fi': 'finnish', 'ru': 'russian', 'sv': 'swedish'}
    doc1 = nlp1(lpar)
    doc2 = nlp2(rpar)
    if args.splitpars:
        tokenized1_curr = [[t.text for t in s.words] for s in doc1.sentences]
        tokenized2_curr = [[t.text for t in s.words] for s in doc2.sentences]
    else:
        tokenized1_curr = [t.text for s in doc1.sentences for t in s.words]
        tokenized2_curr = [t.text for s in doc2.sentences for t in s.words]
    tok1.append(tokenized1_curr)
    tok2.append(tokenized2_curr)
    if args.nonlemmatized:
        lem1.append(tokenized1_curr)
        lem2.append(tokenized2_curr)
    else:
        if args.splitpars:
            lem1.append([[t.lemma for t in s.words] for s in doc1.sentences])
            lem1.append([[t.lemma for t in s.words] for s in doc2.sentences])
        else:
            lem1.append([t.lemma for s in doc1.sentences for t in s.words])
            lem2.append([t.lemma for s in doc2.sentences for t in s.words])

    if len(tok1) >= 50:
        if not(args.resume and num_entries+50 < already_processed):
            try:
                emb1 = elmo1.embed_batch(tok1)
                emb2 = elmo2.embed_batch(tok2)
            except:
                emb1 = []
                emb2 = []
                for bpart in range(10):
                    emb1.append(elmo1.embed_batch(tok1[bpart*5:bpart*5+5]))
                    emb2.append(elmo2.embed_batch(tok2[bpart*5:bpart*5+5]))


        for s in range(len(lem1)): # for each sentence in paragraph
            for w in range(len(lem1[s])): # for each word in sentence
                word1 = lem1[s][w]
                if word1 in appearances and appearances[word1] >= args.limit: 
                    continue # word w already found in args.limit number of contexts
                if word1 in dictionary and any(dw in lem2[s] for dw in dictionary[word1]):
                    for transl in dictionary[word1]:
                        if lem1[s].count(word1) != 1 or lem2[s].count(transl) != 1:
                            continue # skip if either L1 word or L2 word doesn't appear exactly once in sentence
                        else:
                            w2 = lem2[s].index(transl) # find matching word from L2
                            if word1 in appearances:
                                appearances[word1] += 1
                            else:
                                appearances[word1] = 1
                            num_entries += 1
                            if num_entries > args.hardmax:
                                print('Max limit reached. Quitting.')
                                break
                            if args.resume and num_entries < already_processed:
                                continue
                            for layer in [0,1,2]:
                                writebuffer.append((emb1[s][layer][w], emb2[s][layer][w2], layer, lem1[s][w], lem2[s][w2], num_entries))
                            if len(writebuffer) >= 600:
                                batch_condict_write(writebuffer, args)
                                writebuffer = []
        tok1 = []
        tok2 = []
        lem1 = []
        lem2 = []
if args.nocompress:
    file_dict.close()
    for l in range(3):
        lang1[l].close()
        lang2[l].close()
