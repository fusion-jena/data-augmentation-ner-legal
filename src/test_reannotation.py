# -*- coding: utf-8 -*-
import sys
import flair.datasets
import fasttext
from flair.data import Sentence
from aug import utils
from somajo import SoMaJo
sys.path.append('../')
from aug import utils

dataset = (utils.loadCorpus("datasets/___1.0"))

#only for testing
def removeTags(sentence):
    newSentence = utils.tokenize(sentence.to_plain_string())
    for token in newSentence:
        token.add_tag('ner', "O")
    return newSentence

#only for testing
def sameTags(originalSentence, newSentence):
    if len(originalSentence) != len(newSentence):
        return False

    for i in range(0,len(originalSentence)):
        if newSentence[i].get_tag('ner').value != originalSentence[i].get_tag('ner').value:
            return False

    return True

def annotateEntities(sentence, entities):
    for entity in entities:

        entityType = entity[1]
        tokenizedEntity = utils.tokenize(entity[0])
        occurences = findStartIndex(tokenizedEntity, sentence)

        if(len(occurences) == 0):
            raise KeyError('entity was not found in back-translated sentence - Error 2')

        for startIndex in occurences:
            sentence = utils.annotateEntity(sentence, startIndex, entityType, len(tokenizedEntity), "", debug=False)

    return sentence

def fullLengthMatch(startIndex, tokenSequence, sentence): #here start index refers to the index at which the sequence to check starts in the sentence
    if startIndex + len(tokenSequence) - 1 > len(sentence) - 1:
        return 
        
    for i in range(0, len(tokenSequence)):
        if not sentence[startIndex + i].text == tokenSequence[i].text:
            return False
    return True

def fullLengthTokenMatch(startIndex, label, length, sentence): #here start index refers to the index at which the sequence to check starts in the sentence
    if startIndex + length - 1 > len(sentence) - 1:
        return False

    for i in range(0, length):
        if "-" in sentence[startIndex + i].get_tag('ner').value: 
            if i == 0:
                if sentence[startIndex + i].get_tag('ner').value != "B-" + label:
                    return False
            if i != 0:
                if sentence[startIndex + i].get_tag('ner').value != "I-" + label:
                    return False
        else:
            return False
    return True

def findStartIndex(tokenizedEntity, sentence):
    startIndices = []
    for i in range(0, len(sentence)):
        if sentence[i].text == tokenizedEntity[0].text:
            if fullLengthMatch(i, tokenizedEntity, sentence):
                startIndices.append(i)
    return startIndices

def ContainsSameSequenceDifferentLabel(entities, sentence): #requires entites list from getEntities() -> [entityString, entityTag, entityStart]
    sequenceMultiplicity = False
    allSameLabel = True
    for entity in entities:
        tokenizedEntity = utils.tokenize(entity[0])
        occurences = findStartIndex(tokenizedEntity, sentence)
        if len(occurences) >= 2:
            sequenceMultiplicity = True
            for startIndex in occurences:
                if not fullLengthTokenMatch(startIndex, entity[1],len(tokenizedEntity), sentence):
                    allSameLabel = False

    return sequenceMultiplicity and not allSameLabel

min = 99999
fails = 0

failedSentences = []
filteredSentences = []
filteredButWorked = []

for idx in range(0,len(dataset.train)):
    sentence = dataset.train[idx]
    entities = utils.getEntities(sentence)

    if (len(sentence) != len(utils.tokenize(sentence.to_plain_string()))):
        continue

    entityList = [entity[0] for entity in entities]

    if len(entities) != 0:
        if ContainsSameSequenceDifferentLabel(entities, sentence):
            filteredSentences.append(sentence)

            try:
                cleanSentence = removeTags(sentence)
                NewSentence = annotateEntities(cleanSentence, entities)

                if not sameTags(sentence, NewSentence):
                    fails += 1
                    failedSentences.append(sentence)
                else:
                    filteredButWorked.append(sentence)
            except:
                pass
        else:

            try:
                cleanSentence = removeTags(sentence)
                NewSentence = annotateEntities(cleanSentence, entities)
                if not sameTags(sentence, NewSentence):
                    fails += 1
                    print("\n\nERROR: (" + str(idx) + ")")
                    print(sentence.to_tagged_string())
                    print(NewSentence.to_tagged_string())
                    failedSentences.append(sentence)
                    
            except:
                raise Exception("SENTENCE SLIPPED")


for tmp in filteredButWorked:
    print()
    print(tmp.to_tagged_string())
        

print("\nFiltered / FilteredButWorked / Fails / Total:")
print(str(len(filteredSentences)) + " " + str(len(filteredButWorked)) + " " + str(fails)  + " "+ str(len(dataset.train)))
print()