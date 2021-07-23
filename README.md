# ELMoGAN
Crosslingual alignment/mapping of ELMo embeddings

This repository provides the code for creation of a dataset used for crosslingual mapping of contextual ELMo embeddings (a so called, contextual dictionary), and the code for non-linear crosslingual mapping with such a dataset. A short description for both is provided below, for more details, please refer to our article Cross-lingual alignments of ELMo contextual embeddings (https://arxiv.org/abs/2106.15986).

## Dataset creation
For dataset creation you need a bilingual dictionary in a tsv format, where each entry is made of two words, the original word in language 1 and its translation in language 2. The two words are tab separated, and each entry is in its own line. You also need a parallel corpus for the two languages, in Moses format.

To create the dataset for the first (non-contextual CNN) layer of embeddings, run `context_dict/get_layer0_embs.py`. For the second and third (contextual LSTM) layers of embeddings, run `context_dict/contextual_dictionary_moses_stanza.py` to tokenize and lemmatize the corpus on the fly, using Stanza tool. You can greatly speed-up the dataset creation by using a pre-lemmatized version of the corpus (if it's available, or lemmatize it yourself). In this case, run `context_dict/contextual_dictionary_moses_prelemmatized.py` instead, providing both the original and the lemmatized corpus.

## ELMoGAN mapping method
To train the cross-lingual non-linear mapping model, run `mapping/map_elmo_bigan-config2.py`, providing the datasets created with the scripts in `context_dict` as described above.

## Mapping usage
The trained model simultaneously maps from language 1 to language 2 and from language 2 to language 1. It expects two vectors on input and outputs two vectors. The first input is an embedding of a word from language 1 and the second input is an embedding of a word from language 2. On the output, the first output vector will be the first input vector mapped to language 2, the second output vector will be the second input vector mapped to language 1.

When mapping from language 1 to 2, provide your embedding as the first vector, and set an arbitrary second vector (e.g. a copy of the embedding or a zero vector). The mapped embedding will then be the first output vector. For mapping from language 2 to 1, just input your embedding as the second vector and assume the second output vector.

