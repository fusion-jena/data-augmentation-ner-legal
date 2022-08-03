# -*- coding: utf-8 -*-

"""### initialise the workspace"""
import time
import os
from os import environ
import re
import sys
import time
import traceback
import random
from math import floor
from datetime import datetime
from pathlib import Path

import fasttext as fasttext
from fasttext import util
import flair.datasets
import mysql.connector
from transformers import pipeline


from BackTranslation import BackTranslation #optional, for backtranslateGoogleOld (free)
from flair.data import Sentence
from flair.data import Corpus
from flair.datasets import ColumnCorpus

from google.cloud import translate
import deepl 
from gensim.models import KeyedVectors
import requests
from bs4 import BeautifulSoup
from somajo import SoMaJo
from tqdm import tqdm

sys.path.append('../')
try: 
    os.makedirs("./logs/")
except FileExistsError:
    pass

"""### general functions"""

#def cosine_similarity(x1, x2):
#  return np.round(np.dot(x1, x2)/(np.linalg.norm(x1)*np.linalg.norm(x2)),5)

def log(message, filename):
    fo = open("logs/" + filename + ".txt", "a")
    fo.write(message + "\n")
    fo.close()

def getIndicesByClass(sentence, logfilename, debug=False):
    entityIndices = []
    otherIndices = []
    skipped = []
    for i in range(0,len(sentence)):
        if re.fullmatch(r"[a-zA-ZäöüÄÖÜß]+", sentence[i].text):
            if (sentence[i].get_tag('ner').value == "O"):
                otherIndices.append(i)
            else:
                entityIndices.append(i)
        else:
            skipped.append(sentence[i].text)
    
    log("\nSkipping (regex mismatch):\n" + str(skipped) + "\n", logfilename)
    return entityIndices, otherIndices

def findStartIndex(tokenizedEntity, sentence):
    startIndices = []
    for i in range(0, len(sentence)):
        if sentence[i].text == tokenizedEntity[0].text:
            if fullLengthMatch(i, tokenizedEntity, sentence):
                startIndices.append(i)
    return startIndices
    
def fullLengthMatch(startIndex, tokenSequence, sentence): #here start index refers to the index at which the sequence to check starts in the sentence
    if startIndex + len(tokenSequence) - 1 > len(sentence) - 1:
        return False
        
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

tokenizer = SoMaJo("de_CMC", split_sentences=False, split_camel_case=False)

def tokenize(sentence): #expects string as input
    if type(sentence) != type("he"):
        print(type(sentence))
        raise TypeError()
    sentences = tokenizer.tokenize_text([sentence]) #expects list of strings as input
    tokenizedSentence = []
    for sen in sentences:
        for token in sen:
            tokenizedSentence.append(str(token))

    return Sentence(tokenizedSentence)

def ContainsSameSequenceDifferentLabel(entities, sentence): #requires entites list from getEntities() -> [entityString, entityTag, entityStart]
    sequenceMultiplicity = False
    allSameLabel = True
    for entity in entities:
        tokenizedEntity = tokenize(entity[0])
        occurences = findStartIndex(tokenizedEntity, sentence)
        #print(entity, occurences)
        if len(occurences) >= 2:
            sequenceMultiplicity = True
            for startIndex in occurences:
                if not fullLengthTokenMatch(startIndex, entity[1],len(tokenizedEntity), sentence):
                    allSameLabel = False

    return sequenceMultiplicity and not allSameLabel

def getFasttextReplacements(model, tokenToReplace, logfilename, debug=False):
    candidates = []
    values = model.get_nearest_neighbors(tokenToReplace, k = 4)
    values = [term[1] for term in values]
    candidates = filterReplacements(values, tokenToReplace, logfilename, debug=debug)
    return candidates

def getThesaurusReplacements(model, tokenToReplace, logfilename, debug=False):

    if type(model) != type("string"):
        word = tokenToReplace
        select_stmt = "SELECT term.word FROM term, synset, term term2 WHERE synset.is_visible = 1 AND synset.id = term.synset_id AND term2.synset_id = synset.id AND term2.word = '" + word + "'"
        model.execute(select_stmt)
        rows = model.fetchall()
        values = []
        for row in rows:
            values.append(row[0])
    else:
        candidates = []
        time.sleep(1)
        x = requests.get("https://www.openthesaurus.de/synonyme/search?q=" + tokenToReplace + "&format=text/xml")
        soup = BeautifulSoup(x.content, 'html.parser')
        values = soup.find_all('term')
        values = [term["term"] for term in values]
    candidates = filterReplacements(values, tokenToReplace, logfilename, debug=debug)
    return candidates

def getWord2VecReplacements(model, tokenToReplace, logfilename, debug=False):
    candidates = []
    try:
        values = model.most_similar(tokenToReplace, topn = 4)
        values = [term[0] for term in values]
        candidates = filterReplacements(values, tokenToReplace, logfilename, debug=debug)
        return candidates
    except:
        log("\nError in: Word2VecReplacements, tokenToReplace: " + tokenToReplace, logfilename)
        return []

def getGloVeReplacements(model, tokenToReplace, logfilename, debug=False):
    candidates = []
    try:
        values = model.most_similar(tokenToReplace, topn = 4)
        values = [term[0] for term in values]
        candidates = filterReplacements(values, tokenToReplace, logfilename, debug=debug)
        return candidates
    except:
        log("\nError in: GloVeReplacements, tokenToReplace: " + tokenToReplace, logfilename)
        return []

def filterReplacements(values, tokenToReplace, logfilename, debug=False):
    filtered = []
    for candidate in values:
        if candidate.lower() == tokenToReplace.lower():
            if debug:
                log("discarded (is original): " + candidate, logfilename)
        elif re.fullmatch(r"[a-zA-ZäöüÄÖÜß]+", candidate):
            filtered.append(candidate)
        else:
            if debug:
                log("discarded (regex): " + candidate, logfilename)
    return filtered

def getCLMReplacements(model, sentence, tokenToReplace, tokenStartIndex, logfilename, debug=False):
    sentence = replaceSpan(sentence, Sentence(tokenToReplace), Sentence(["<mask>"]), tokenStartIndex, logfilename, debug=debug)
    newPlainSentence = sentence.to_plain_string() + " </s>" #only for roBERTa
    if debug:
        log("replaced " + tokenToReplace + " with <mask>", logfilename)
        log(newPlainSentence, logfilename)
    res = model(newPlainSentence)
    values = [dic["token_str"] for dic in res]
    candidates = filterReplacements(values, tokenToReplace, logfilename, debug=debug)
    return candidates

def getReplacement(sentence, plainSentence, tokenToReplace, tokenStartIndex, modelname, model, logfilename, deterministic=True, debug=False):
    if modelname == "fasttext":
        candidates = getFasttextReplacements(model, tokenToReplace,logfilename, debug=debug)
    elif modelname == "word2vec":
        candidates = getWord2VecReplacements(model, tokenToReplace,logfilename, debug=debug)
    elif modelname == "glove":    
        candidates = getGloVeReplacements(model, tokenToReplace,logfilename, debug=debug)
    elif modelname == "clm":     
        candidates = getCLMReplacements(model, sentence, tokenToReplace, tokenStartIndex, logfilename, debug=debug)
    elif modelname == "thesaurus":
        candidates = getThesaurusReplacements(model, tokenToReplace, logfilename, debug=debug)
    else:
        raise Exception("select valid model:\nfasttext, word2vec, glove, distilroberta-base")

    if debug:
        log("candidates for <" + tokenToReplace + "> :" + str(candidates), logfilename)
    if len(candidates) == 0:
        return -1

    if len(candidates) >= 2 and not deterministic:
        if random.random() < 0.7:
            return candidates[0]
        else:
            return candidates[1]

    return candidates[0]

def combineCorpora(corpusA, corpusB, dev, test, dirname, timestamp=True):
    #saves as train, creates dummy test and dev sets
    if timestamp:
        current_time = datetime.now().strftime("%H_%M_%S")
        dirname = dirname + "_" + current_time
    try: 
        os.makedirs("./"+ dirname + "/")
    except FileExistsError:
        print("Directory " , dirname ,  " already exists") 
    
    with open("./" + dirname + "/train.txt", "w") as myfile:
        for sentence in corpusA:
            for token in sentence:
                myfile.write(token.text + " " + token.get_tag('ner').value + "\n")
            myfile.write("\n")
        for sentence in corpusB:
            for token in sentence:
                myfile.write(token.text + " " + token.get_tag('ner').value + "\n")
            myfile.write("\n")
    with open("./" + dirname + "/test.txt", "w") as myfile:
        for sentence in test:
            for token in sentence:
                myfile.write(token.text + " " + token.get_tag('ner').value + "\n")
            myfile.write("\n")
    with open("./" + dirname + "/dev.txt", "w") as myfile:
        for sentence in dev:
            for token in sentence:
                myfile.write(token.text + " " + token.get_tag('ner').value + "\n")
            myfile.write("\n")

def saveCorpus(train, dev, test, dirname, timestamp=True):
    #saves as train, creates dummy test and dev sets
    if timestamp:
        current_time = datetime.now().strftime("%H_%M_%S")
        dirname = dirname + "_" + current_time
    try: 
        os.makedirs("./"+ dirname + "/")
    except FileExistsError:
        print("Directory " , dirname ,  " already exists, skipping...") 
        return
    
    with open("./" + dirname + "/train.txt", "w") as myfile:
        for sentence in train:
            for token in sentence:
                myfile.write(token.text + " " + token.get_tag('ner').value + "\n")
            myfile.write("\n")
    with open("./" + dirname + "/test.txt", "w") as myfile:
        for sentence in test:
            for token in sentence:
                myfile.write(token.text + " " + token.get_tag('ner').value + "\n")
            myfile.write("\n")
    with open("./" + dirname + "/dev.txt", "w") as myfile:
        for sentence in dev:
            for token in sentence:
                myfile.write(token.text + " " + token.get_tag('ner').value + "\n")
            myfile.write("\n")

def loadCorpus(path):
    columns = {0: 'text', 1: 'ner'}
    return ColumnCorpus(path, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt')

def initSource(source):
    if source == "fasttext":
        #fasttext (takes 40min, 4GB), source!
        fasttext.util.download_model('de', if_exists='ignore')  # German
        return fasttext.load_model('cc.de.300.bin')
    elif source == "word2vec-en":
        #1500mb source?
        cmd = "!wget -P . -c \"https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz\""
        os.system(cmd)
        return KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin.gz", binary=True)
    elif source == "word2vec-de":
        #700mb https://github.com/devmount/GermanWordEmbeddings
        cmd = '!wget -P . -c "https://cloud.devmount.de/d2bc5672c523b086/german.model"'
        os.system(cmd)
        return KeyedVectors.load_word2vec_format("german.model", binary=True)
    elif source == "glove":
        #glove (takes 5min, 370MB) CURRENTLY ENGLISH lowercase ONLY
        return api.load("glove-wiki-gigaword-300")
    elif source == "clm":
        return pipeline('fill-mask', model="xlm-roberta-large")
    elif source == "thesaurus":
        print("initialization of thesaurus is included in SynonymReplacement function")
    else:
        print("please select valid source: fasttext, word2vec, glove, clm")

def getEntities(sentence):
    #returns array containing all entities in the sentence 
    entitiesInternal = []
    currentEntity = ""
    lastEntityTag = ""
    lastEntityStart = ""
    for token in sentence:
        tag = token.get_tag('ner') 
        if tag.value == "O":
            pass
        if tag.value[0] == "B" or tag.value[0] == "S":
            if currentEntity != "":
                entitiesInternal.append([currentEntity, lastEntityTag, lastEntityStart])
            lastEntityTag = tag.value.split("-")[1]
            lastEntityStart = token.idx - 1
            currentEntity = token.text
        if (tag.value[0] == "I") or (tag.value[0] == "E"):
            currentEntity = currentEntity + " " + token.text
    if currentEntity != "":
            entitiesInternal.append([currentEntity, lastEntityTag, lastEntityStart])
    return entitiesInternal

def getRandomEntity(entity_type, dictionary):
    return random.sample(list(dictionary[entity_type]), 1)[0]

def buildEntityDict(corpus, verbose=False):
    #double check if this works correctly, looks like there are a lot of mistakes in the labels??
    entities = {}
    for sentence in corpus:
        entitiesInternal = []
        currentEntity = ""
        lastEntityTag = ""
        for token in sentence:
            tag = token.get_tag('ner') 
            if tag.value == "O":
                pass
            if tag.value[0] == "B" or tag.value[0] == "S":
                if currentEntity != "":
                    entitiesInternal.append((currentEntity, lastEntityTag))
                    if lastEntityTag in entities:
                        entities[lastEntityTag].add(currentEntity)
                    else:
                        entities[lastEntityTag] = {currentEntity}
                lastEntityTag = tag.value.split("-")[1]
                currentEntity = token.text
            if (tag.value[0] == "I") or (tag.value[0] == "E"):
                currentEntity = currentEntity + " " + token.text
        if currentEntity != "":
                entitiesInternal.append((currentEntity, lastEntityTag))
                if lastEntityTag in entities:
                    entities[lastEntityTag].add(currentEntity)
                else:
                    entities[lastEntityTag] = {currentEntity}
        if verbose:
            print("\nextracted : " + str(entitiesInternal) + "\nfrom:  " + sentence.to_tagged_string())
    if verbose:
        for cls in entities:
            print(cls +  ": " + str(entities[cls]))
    return entities

def replaceSpan(sentence, tokenizedOriginalEntity, tokenizedNewEntity, entityStartIndex,logfilename, debug=False):
    tokenizedSentence = []
    for token in sentence:
        tokenizedSentence.append(token.text)

    #remove old entity
    removals = []
    for i in range(0,len(tokenizedOriginalEntity)):
        removal = tokenizedSentence.pop(entityStartIndex) #do not add i because array automatically shifts to the left 
        removals.append(removal)

    #insert new entity
    insertions = []
    for i in range(0, len(tokenizedNewEntity)):
        tokenizedSentence.insert(entityStartIndex + i , tokenizedNewEntity[i].text)
        insertions.append(tokenizedNewEntity[i].text)

    if debug:
        log("tokenized sentence: " + str(tokenizedSentence), logfilename)
        log("removed tokens:     " + str(removals), logfilename)
        log("inserted tokens:    " +str(insertions), logfilename)
    return Sentence(tokenizedSentence) #pretokenized input (list of string elements) as input

#wrapper function
def annotateEntities(sentence, entities, debug, logfilename):
    foundBacktranslated = 0
    foundOriginal = 0

    for entity in entities:

        entityType = entity[1]
        tokenizedEntityOriginal = tokenize(entity[0])
        tokenizedAlternativeEntity = tokenize(entity[3])

        occurencesOriginal = findStartIndex(tokenizedEntityOriginal, sentence)
        occurencesAlternative = findStartIndex(tokenizedAlternativeEntity, sentence)

        if len(occurencesAlternative) == 0 and len(occurencesOriginal)  == 0:
            raise KeyError('entity was not found in back-translated sentence - Error 1')

        if len(occurencesOriginal) >= len(occurencesAlternative):
            tokenizedEntity = tokenizedEntityOriginal
            occurences = occurencesOriginal
            foundOriginal += 1
        else:
            tokenizedEntity = tokenizedAlternativeEntity
            occurences = occurencesAlternative
            foundBacktranslated += 1

        if debug:
            log("targetEntity: " + str(tokenizedEntity[0].text), logfilename)
            log("entityStart: " + str(occurences), logfilename)
            log("entityLength: " + str(entityLength), logfilename)

        for startIndex in occurences:
            sentence = annotateEntity(sentence, startIndex, entityType, len(tokenizedEntity), "", debug=False)

    return sentence, foundOriginal, foundBacktranslated

def annotateBeforeEntity(sentence, augmentedSentence, entityStartIndex,logfilename, debug=False):              
    for i in range(0,entityStartIndex):
        augmentedSentence[i].add_tag('ner', sentence[i].get_tag('ner').value)
        if debug:
            log("(copy) annotated: " + augmentedSentence[i].text + " " + sentence[i].get_tag('ner').value, logfilename)
    return augmentedSentence

def annotateEntity(augmentedSentence, entityStartIndex, entityType, entityTokenLength, logfilename, debug=False):
    #manually annotate the tokens part of the entity
    if (entityTokenLength == 1): #can replace this with S to have IOBES tagged sentences as output
        augmentedSentence[entityStartIndex].add_tag('ner', "B-" + entityType)
        if debug:
            log("tokenlength 1 -> annotated S " + entityType, logfilename)
            log("(manual) annotated: " + augmentedSentence[entityStartIndex].text + " "  + "B-" + entityType, logfilename)
    else:
        if debug:
            log("tokenlength > 1 -> annotated others " +  entityType, logfilename)
        for i in range(0,entityTokenLength):
            if i == 0:
                augmentedSentence[entityStartIndex + i].add_tag('ner', "B-" + entityType)
                if debug:
                    log("(manual) annotated: " + augmentedSentence[entityStartIndex + i].text + " " + "B-" + entityType, logfilename)
            elif i == entityTokenLength-1: #can replace this with E to have IOBES tagged sentences as output
                augmentedSentence[entityStartIndex + i].add_tag('ner', "I-" + entityType)
                if debug:
                    log("(manual) annotated: " + augmentedSentence[entityStartIndex + i].text + " " + "I-" + entityType, logfilename)
            else:
                augmentedSentence[entityStartIndex + i].add_tag('ner', "I-" + entityType)
                if debug:
                    log("(manual) annotated: " + augmentedSentence[entityStartIndex + i].text + " " + "I-" + entityType, logfilename)
    return augmentedSentence

def annotateNonEntity(augmentedSentence, entityStartIndex, entityTokenLength, logfilename, debug=False):
    for i in range(0,entityTokenLength):
        augmentedSentence[entityStartIndex + i].add_tag('ner', "O")
        if debug:
            log("(manual) annotated: O ", logfilename)
    return augmentedSentence

def annotateAfterEntity(sentence, augmentedSentence, entityEndIndex, offset,logfilename, debug=False):
    if debug:
        log("iterating until i = " + str(len(sentence)-1), logfilename)
    for i in range(entityEndIndex,len(sentence)):
        if debug:
            log("(copy) annotated: " + augmentedSentence[i+offset].text + " " + sentence[i].get_tag('ner').value, logfilename)
            log("new: " + str(augmentedSentence[i+offset]) +" "+ str(i+offset)          + "/" + str(len(augmentedSentence)-1), logfilename)
            log("old: " + str(sentence[i])   +" "+ str(i) + "/" + str(len(sentence)-1) + "\n====", logfilename)
        augmentedSentence[i+offset].add_tag('ner', sentence[i].get_tag('ner').value)
    return augmentedSentence

""" def plotClassFrequencies(fractions):
    data = {}
    #save the lists 
    for key in fractions.keys():
        dataset = fractions[key]
        dictionary = buildEntityDict(dataset)
        for cls in dictionary:
            if cls in data:
                data[cls].append(len(dictionary[cls]))
            else:
                data[cls] = [len(dictionary[cls])]
        
    df = pd.DataFrame(data, index=['10%', '30%', '50%', '70%', '90%','100%'])
    df = df.div(df.sum(axis=1), axis=0) #convert counts to percentages

    df.plot(kind='bar', stacked=True)
    plt.xlabel('Datasets')
    plt.ylabel('label frequency [%]')
    plt.title('label frequency by class across datasets') """