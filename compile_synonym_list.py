from __future__ import print_function

import argparse
import sys
import os
import nltk
import spacy
from collections import defaultdict
import csv
import pandas
import codecs
from nltk.corpus import wordnet as wn
from pattern.en import lexeme, singularize, pluralize, comparative, superlative, conjugate, tenses
from scipy import sparse
import pickle
from nltk.corpus import wordnet as wn

parser = argparse.ArgumentParser(description='Build synonym information about vocabulary set')
parser.add_argument('--data_path', type=str, default='../data/wiki_polyglot/no_oovs/shuffled',
                    help='location of the data corpus')
args = parser.parse_args()

vocab_df = pandas.read_csv(os.path.join(args.data_path, "vocab.csv"), index_col=0, encoding='utf-8')

wordnet_types = {"NOUN":wn.NOUN, "VERB":wn.VERB, "ADV":wn.ADV, "ADJ":wn.ADJ}
wordnet_pos_tags = {v:k for k,v in wordnet_types.items()}

def match_all_inflections(source_word, target_words, pos):
    if pos == wn.VERB:
        inflections = set()
        conjugations = tenses(source_word)
        for tense,person,number,mood,aspect in conjugations:
            inflections.update([conjugate(word, tense=tense,person=person,number=number,mood=mood,aspect=aspect) for word in target_words])
        return inflections
    elif pos == wn.NOUN:
        return [singularize(word) for word in target_words] + [pluralize(word) for word in target_words]
    else: # pos == "ADJ" or pos == "ADV"
        return target_words

def all_inflected_synonyms(word, stanford_pos=None):
    pos = None
    if stanford_pos is not None and stanford_pos in wordnet_types:
        pos = wordnet_types[stanford_pos]

    synonyms = {v:set() for v in wordnet_types.keys()}
    try:
        synsets = wn.synsets(word, pos=pos)
    except:
        print(word)
        return {}
    for synset in synsets:
        target_pos = synset.name().split('.')[1]
        if pos is not None and target_pos != pos:
            continue  # only include synonyms for correct part of speech
        if target_pos not in wordnet_pos_tags:
            continue  # noun, adj, adv, verb only

        target_lemmata = [l.name().lower() for l in synset.lemmas() if '_' not in l.name()]
        inflected_synonyms = [x for x in match_all_inflections(word, target_lemmata, target_pos) if x in vocab_df.index]
        synonyms[wordnet_pos_tags[target_pos]].update(inflected_synonyms)

    return synonyms

def has_wordnet_synonyms(word):
    if token not in vocab_df.index:
        return False
    for pos,count in vocab_df.loc[token].iteritems():
        if count > 100:
            return True  # token is a noun, verb, adj, or adv

print('Building synonym set')
pydict_types = {'Adjective': 'ADJ', 'Noun': 'NOUN', 'Adverb': 'ADV', 'Verb': 'VERB'}

from PyDictionary import PyDictionary
dictionary=PyDictionary()

pydict_thesaurus = {}

def all_inflected_synonyms_pydictionary(word):
    try:
        dictionary_meaning = dictionary.meaning(word)
        dictionary_synonym = dictionary.synonym(word)
    except:
        return {}
    if dictionary_meaning is None or dictionary_synonym is None:
        return {}
    pos_tags = [t for t in dictionary_meaning.keys() if t in pydict_types]

    synonyms = {pydict_types[t]:set() for t in pos_tags}
    for synonym in dictionary_synonym:
        target_meaning = dictionary.meaning(synonym)
        if target_meaning is None:
            continue
        for target_pos in target_meaning.keys():
            if target_pos not in pos_tags:
                continue  # only include synonyms for correct part of speech

            inflected_synonyms = [x for x in match_all_inflections(word, [synonym], wordnet_types[pydict_types[target_pos]]) if x in vocab_df.index]
            synonyms[pydict_types[target_pos]].update(inflected_synonyms)

    return synonyms

with codecs.open(os.path.join(args.data_path, "vocab_pydict.csv"), 'w', encoding='utf-8') as thesaurus_fh:
    for token in vocab_df.index:
        if not has_wordnet_synonyms(token):
            continue
        print(token, '\t', all_inflected_synonyms_pydictionary(token), file=thesaurus_fh)

print('Building wordnet similarity database')

import ast

pos_tag_sets = {tag:set([word2idx[w] for w,cnt in vocab_df[tag].items()\
                         if cnt > 100])\
                for tag in ['ADJ', 'ADV', 'NOUN', 'VERB']}

words_of_interest = set().union(*[v for k,v in pos_tag_sets.items()])

def path_similarity(word1, word2):
    idx1 = word2idx[word1]
    idx2 = word2idx[word2]
    
    same_pos = False
    for tag in pos_tag_sets.keys():
        if idx1 in pos_tag_sets[tag] and idx2 in pos_tag_sets[tag]:
            same_pos = True
            break
    if not same_pos:
        return None
    
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word1)
    
    # return minimum similarity
    minimum_similarity = None
    for left_synonym in synsets1:
        for right_synonym in synsets2:
            similarity = left_synonym.path_similarity(right_synonym) 
            if similarity is not None:
                if minimum_similarity is None or minimum_similarity < similarity:
                    minimum_similarity = similarity
    return minimum_similarity

def compute_wordnet_similarities(outfile):
    for idx_1, word_1 in vocab_df.index[:-1]:
        if idx_1 % 500 == 0:
            print('..', end='')
        for idx_2, word_2 in vocab_df.index[idx_1+1:]:
            if word_1 not in words_of_interest or word_2 not in words_of_interest:
                continue
            similarity = path_similarity(word_1, word_2)
            if similarity is not None:
                print(word_1, '\t', word_2, '\t', similarity, file=outfile)
                
with codecs.open(os.path.join(args.data_path, 'wordnet_similarities.csv'), 'w', encoding='utf-8') as outfile:
    compute_wordnet_similarities(outfile)
