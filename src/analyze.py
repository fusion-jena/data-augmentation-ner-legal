import sys
sys.path.append('../')
from aug import utils

path = str(sys.argv[1])

dataset = (utils.loadCorpus(path))

datasetTrain = dataset.train
datasetDev = dataset.dev
datasetTest = dataset.test

def countClassOccurences(corpus, verbose=False):
    entities = {}
    for sentence in corpus:
        entitiesInternal = []
        currentEntity = ""
        lastEntityTag = ""
        for token in sentence:
            tag = token.get_tag('ner') 
            if tag.value == "O":
                pass

            elif tag.value[0] == "B" or tag.value[0] == "S":
                label = tag.value.split("-")[1]
                if label in entities:
                    entities[label] += 1
                else:
                    entities[label] = 1
            elif (tag.value[0] == "I") or (tag.value[0] == "E"):
                pass

    return entities

dic = countClassOccurences(datasetTrain)

x = 0
y = 0
z = 0
v = 0
hasEntities = 0
characters = 0
irreproducibleTokenization = 0

for sentence in dataset.train:
    characters += len(sentence.to_plain_string())

    if (len(sentence) != len(utils.tokenize(sentence.to_plain_string()))):
        irreproducibleTokenization += 1

    entities = utils.getEntities(sentence)

    if len(entities) != 0:
        hasEntities += 1

    entityStringList = [entity[0] for entity in entities]
    if utils.ContainsSameSequenceDifferentLabel(entities, sentence):
        v += 1


print("there are " + str(characters) + " characters in the training set")
print("hasNestedEntities, hasMultipleEntities, hasBoth, sameSequence, totalSentences")
print(x,y,z,v, len(dataset.train))
print(x/len(dataset.train),y/len(dataset.train),z/len(dataset.train), v/len(dataset.train), 100 )
print("of sentences with entities: ")
print(x,y,z,v, hasEntities)
print(x/hasEntities,y/hasEntities,z/hasEntities, v/hasEntities, 100 )
print()
print("there were " + str(irreproducibleTokenization) + " irreprod. tokenization sentences  (" + str(irreproducibleTokenization/len(dataset.train)) + ")")
print()
print("train-Sentences: ", len(datasetTrain))
print("dev-Sentences: ", len(datasetDev))
print("test-Sentences: ", len(datasetTest))

print("Entities (train): (#unique tags: " + str(len(dic.keys())) + ")")

for tag in dic.keys():
    print(" " + str(tag) + " "+ str(dic[tag]))