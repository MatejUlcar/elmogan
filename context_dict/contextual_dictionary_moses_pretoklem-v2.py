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

# !!! THIS CODE ASSUMES PRETOKENIZED AND PRELEMMATIZED TEXT !!!

    
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
    if args.nonlemmatized:
        with open(ali1, 'r') as leftpars, open(ali2, 'r') as rightpars:
            rline = False
            for line in leftpars:
                lpar = line.strip()
                rpar = rightpars.readline()
                rpar = rpar.strip()
                yield lpar, rpar, lpar, rpar
    else:
        with open(ali1+'.lem', 'r') as leftlem, open(ali2+'.lem', 'r') as rightlem, open(ali1, 'r') as leftpars, open(ali2, 'r') as rightpars:
            for line in leftpars:
                lpar = line.strip()
                rpar = rightpars.readline()
                rpar = rpar.strip()
                llem = leftlem.readline()
                llem = llem.strip()
                rlem = rightlem.readline()
                rlem = rlem.strip()
                yield lpar, rpar, llem, rlem

    
#def find_word_pairs():
#    matchingpars = match_pars(alignfile, ali1, ali2)
#    for lpar, rpar in matchingpars:
#        lpar.replace('</s>', '\n')
#        rpar.replace('</s>', '\n')
        

parser = argparse.ArgumentParser(description='Find words from languages L1 and L2 in same context, output dictionaries and embeddings for those words.')
parser.add_argument('-i', '--input1', help='L1 pretokenized corpus (.tok)')
parser.add_argument('-j', '--input2', help='L2 pretokenized corpus (.tok)')
parser.add_argument('-o', '--output', help='path to output folder')
parser.add_argument('-d', '--dictionary', help='L1-L2 dictionary')
parser.add_argument('-l1', '--lang1', help='L1 2-letter language code')
parser.add_argument('-l2', '--lang2', help='L2 2-letter language code')
#above mandatory, below not/also now
#parser.add_argument('--noembed', action='store_true', help='do not produce embeddings, output just word pairs in context')
parser.add_argument('--options1')
parser.add_argument('--options2')
parser.add_argument('--weights1')
parser.add_argument('--weights2')
parser.add_argument('--cuda', type=int, default=-1)
parser.add_argument('--limit', type=int, default=20, help='max contexts per dictionary entry')
parser.add_argument('--hardmax', type=int, default=3000000, help='hard limit on max number of entries output')
parser.add_argument('--nocompress', action='store_true', help='do not xz compress the output')
parser.add_argument('--nonlemmatized', action='store_true', help='do not match word lemmas, but only actual word forms')
parser.add_argument('--resume', help='resume from given contextual .dict')
parser.add_argument('--elmobatch', type=int, default=50, help='number of lines to be embedded in one batch (must be multiple of --fallbackbatch')
parser.add_argument('--fallbackbatch', type=int, default=5, help='fall back to this size of elmo batch, if the above is too large (rechecked every batch)')
args = parser.parse_args()

#entry_limit = 500000 # hard limit on max number of entries


# LOAD LEMMATIZERS AND EMBEDDERS
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
            #already_appeared = re.split('(\d+)', line)
            #appearances[already_appeared[0].strip()] = int(already_appeared[1])
#print("matching pars:", len(matchingpars))
for lpar, rpar, llem, rlem in matchingpars:
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
    #tok1.append(word_tokenize(lpar, language=expand_lang[args.lang1]))
    #tok2.append(word_tokenize(rpar, language=expand_lang[args.lang2]))
    tok1.append(lpar.split())
    tok2.append(rpar.split())
    lem1.append(llem.split())
    lem2.append(rlem.split())
    #doc1 = nlp1(lpar)
    #doc2 = nlp2(rpar)
    #tok1 = [[word.text for word in sent.words] for sent in doc1.sentences]
    #tok2 = [[word.text for word in sent.words] for sent in doc2.sentences]
    #tok1 = [s.split() for s in lpar.split('\n')]
    #tok2 = [s.split() for s in rpar.split('\n')]
    #lem1.append(lparlem.split())
    #lem2.append(rparlem.split())
#    try:
#        assert (len(tok1[0]) == len(lem1[0])), "Tokenized and lemmatized sentence of lang1 unequal length."
#        assert (len(tok2[0]) == len(lem2)), "Tokenized and lemmatized sentence of lang2 unequal length."
#    except AssertionError as err:
#        print('Sentence skipped:', err)
#        print(tok2)
#        print(lem2)
#        break
    #lem1 = [[word.lemma for word in sent.words] for sent in doc1.sentences]
    #lem2 = [word.lemma for sent in doc2.sentences for word in sent.words]
    if len(tok1) >= args.elmobatch:
        if not(args.resume and num_entries+args.elmobatch < already_processed):
            try:
                emb1 = elmo1.embed_batch(tok1)
                emb2 = elmo2.embed_batch(tok2)
            except:
                emb1 = []
                emb2 = []
                print('Embedding failed, trying with smaller batch')
                for bpart in range(args.elmobatch//args.fallbackbatch):
                    emb1.append(elmo1.embed_batch(tok1[bpart*args.fallbackbatch:(bpart+1)*args.fallbackbatch]))
                    emb2.append(elmo2.embed_batch(tok2[bpart*args.fallbackbatch:(bpart+1)*args.fallbackbatch]))
        #emb2_flat = flatten_emb(emb2)
        #emb2 = None
        #print(len(tok1), tok1)
        #print(len(tok2), tok2)
        #print(len(lem1), lem1)
        #print(len(lem2), lem2)

        for s in range(len(lem1)): # for each sentence in paragraph
            #print(s)
            for w in range(len(lem1[s])): # for each word in sentence
                word1 = lem1[s][w]
                #print(word1, word1 in dictionary, [dw in lem2 for dw in dictionary[word1]], dictionary[word1], lem2)
                if word1 in appearances and appearances[word1] >= args.limit: 
                    continue # word w already found in args.limit number of contexts
                if word1 in dictionary and any(dw in lem2[s] for dw in dictionary[word1]):
                    #print("in dict:",word1, dictionary[word1])
                    for transl in dictionary[word1]:
                        if lem1[s].count(word1) != 1 or lem2[s].count(transl) != 1:
                            #print(lem1, lem2, word1, transl) 
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
                                #add2condict(emb1[s][layer][w], emb2_flat[layer][w2], layer, lem1[s][w], lem2[w2], num_entries)
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
