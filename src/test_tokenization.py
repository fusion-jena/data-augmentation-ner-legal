# -*- coding: utf-8 -*-
import sys
import flair.datasets
from flair.data import Sentence
from aug import utils
from somajo import SoMaJo
sys.path.append('../')
from aug import utils

path = str(sys.argv[1])
dataset = (utils.loadCorpus(path))
corpus = dataset.train

tokenizer = SoMaJo("de_CMC", split_sentences=False, split_camel_case=True)

count = 0
for i in range(0,len(corpus)):
    sentence = corpus[i]

    sentences = tokenizer.tokenize_text([sentence.to_plain_string()]) #expects list of strings as input
    tokenizedSentence = []
    for sen in sentences:
        for token in sen:
            tokenizedSentence.append(str(token))

    tokenizedSentence = Sentence(tokenizedSentence) #flair tokenizer supplied with list uses the pretokenized sequence
    #tokenizedSentence = Sentence(sentence.to_plain_string())

    if (len(sentence) != len(tokenizedSentence)):
        count += 1


print()
print("incorrectly tokenized sentences: (abs, perc)")
print(count, count/len(dataset.train))
print(len(dataset.train))

# citation somajo https://aclanthology.org/W16-2607/

# 23847 0.3574, with flair default tokenizer (SegTok)
# 6679 0.1001 issues, with tokenize_text somajo split_sentences = true, split_camel_case = true
# 6051 0.0907 issues, with tokenize_text somajo split_sentences = false, split_camel_case = false #this is the accurate configuration
# 6051 0.0907 issues, with tokenize_text somajo split_sentences = true, split_camel_case = false 
# 6679 0.1001 issues, with tokenize_text somajo split_sentences = false, split_camel_case = true

#this usage of tokenize does not respect the sentence splits, so there is no difference in results
# basically always use "false" -> we do not want sentences to be split into sentences again, we work with the sentence splits we were given