import flair
import sys
import os
sys.path.append('../')
from aug import utils
from tqdm import tqdm
import random


def generateFractions(corpus, verbose=False):
    data = {"0.01": [], "0.1": [], "0.3": [], "0.5": [], "0.7": [], "0.8": [], "0.9": [], "1.0": []}

    # create lists containing the sentences
    for i in range(0, len(corpus.train)):
        keys = [0.01, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
        percent = i/len(corpus.train)
        sentence = corpus.train[i]
        for key in keys:
            if percent < key:
                data[str(key)].append(sentence)

    # save the lists
    for key in data.keys():
        dataset = data[key]
        dirname = "/datasets/___" + str(key)
        utils.saveCorpus(dataset, corpus.dev, corpus.test, dirname, timestamp=False)
        print("Dataset: " + str(key) + "  Size: " + str(len(dataset)))
        dictionary = utils.buildEntityDict(dataset)
        if verbose:
            print("\n" + dirname)
            for cls in dictionary:
                print(cls + " : " + str(len(dictionary[cls])))


def splitTrainTestDev(corpus, testSize, devSize):
    data = {"train": [], "test": [], "dev": []}
    corpus = list(corpus)
    random.shuffle(corpus)

    if testSize + devSize + (1 - testSize - devSize) != 1:
        raise Exception("invalid set sizes provided!")
        return

    try: 
        os.makedirs("./datasets/")
    except FileExistsError:
        print("Warning: Directory datasets already exists, aborting...")
        return 

    # create lists containing the sentences
    for i in range(0, len(corpus)):
        percent = i/len(corpus)
        sentence = corpus[i]
        if percent < testSize:
            data["test"].append(sentence)
        elif percent > testSize and percent < testSize+devSize:
            data["dev"].append(sentence)
        else:
            data["train"].append(sentence)

    # save the lists
    for key in data.keys():
        dataset = data[key]

        with open("./datasets/" + str(key) + ".txt", "w") as myfile:
            for sentence in dataset:
                for token in sentence:
                    myfile.write(token.text + " " + token.get_tag('ner').value + "\n")
                myfile.write("\n")
        print("Dataset: " + str(key) + "  Size: " + str(len(dataset)))


random.seed(0)
flair.set_seed(42)
corpus = (utils.loadCorpus("LER_dataset"))
splitTrainTestDev(corpus.train, 0.15, 0.15)
dataset = (utils.loadCorpus("datasets"))
generateFractions(dataset)
print("finished successfully")
quit()
