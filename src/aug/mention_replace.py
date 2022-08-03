from utils import *
import sys

# use like this:
# python3 aug/mention_replace.py path

#NOTE: generate fractions beforehand, using generate_datasets script

def MentionReplace(dataset, dirname, logfilename="log", debug=False, save=False, timestamp=True, entityDictionary={}):
    corpus = dataset.train
    start_time = time.time()
    logfilename = logfilename + "_MR"
    
    originalCorpus = {"0.01" : [], "0.1": [], "0.3":[], "0.5":[], "0.7":[], "0.8":[], "0.9":[], "1.0":[]}
    augmentedCorpus = {"0.01" : [], "0.1": [], "0.3":[], "0.5":[], "0.7":[], "0.8":[], "0.9":[], "1.0":[]}

    successfulReplacements = failedReplacements = noChangeAugmentations = totalTokens = totalEntities = noEntitySentence = irreproducibleTokenization = 0
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%Z")

    log("MentionReplacement - " +current_time +  "\n\nCONFIGURATION:",logfilename)
    log("corpus-size:         " + str(len(corpus)), logfilename)
    log("corpus-hash:         " + str(hash(corpus)), logfilename)
    log("modelname:           " + "dictionary",logfilename)
    log("modify entities:     " + str(True), logfilename)
    log("modify non-entities: " + str(False),logfilename)
    log("modify-percentage:   " + str(1),logfilename)
    log("debug:               " + str(debug),logfilename)
    log("save:                " + str(save),logfilename)
    log("entity-dictionary:   ",logfilename)
    for cls in entityDictionary:
        log(cls +  ": " + str(entityDictionary[cls]), logfilename)

    for i in tqdm(range(len(corpus))):
        sentence = corpus[i]
        originalSentence = augmentedSentence = sentence
        totalTokens += len(sentence)
        sentenceOffset = 0

         #add original sentence
        keys = [0.01, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
        percent = i/len(corpus)
        for key in keys:
            if percent <= key:
                originalCorpus[str(key)].append(originalSentence)

        #sanity check - if we can reproduce the way it is tokenized, important for further handling of the sentence
        if (len(sentence) != len(tokenize(sentence.to_plain_string()))):
            irreproducibleTokenization += 1
            continue

        log( "\n"+"%%%%%%%%%"*100 + "\n" + "\ntagged original sentence:\n"  + originalSentence.to_tagged_string(), logfilename)
        try:
            entitiesInSentence = getEntities(sentence)

            if len(entitiesInSentence) == 0:
                noEntitySentence += 1
                

            totalEntities += len(entitiesInSentence)
            replacements = [] 

            if debug:
                log("found entities: " + str(entitiesInSentence), logfilename)

            #order matters entities -> other or other -> entities!
            #currently always replaces all entities with another random entity of the same class
            log("Starting replacement of entities." , logfilename)
            for j in range(0, len(entitiesInSentence)):
                
                tokenToReplace, entityType, entityStartIndex = entitiesInSentence[j]
                tokenizedOriginalEntity = tokenize(tokenToReplace)
                candidate = getRandomEntity(entityType, entityDictionary)

                #maybe prevent replacement with itself
                if candidate != -1:      
                    #tokenize the new entity
                    tokenizedNewEntity = tokenize(candidate)
                    #get entity end index in the old sentence
                    entityEndIndexInOriginal = entityStartIndex + len(tokenizedOriginalEntity)
                    #get offset for labels of new and old sentence
                    offset = len(tokenizedNewEntity) - len(tokenizedOriginalEntity)
                    #set sentence to the augmented sentence from last iteration, has labels, no change yet
                    sentence = augmentedSentence
                    #replace old entity with new entity, return sentence without any labels
                    augmentedSentence = replaceSpan(sentence, tokenizedOriginalEntity, tokenizedNewEntity, entityStartIndex,logfilename, debug=debug )
                    #copy labels from original sentence until entity
                    augmentedSentence = annotateBeforeEntity(sentence, augmentedSentence, entityStartIndex,logfilename, debug=debug)
                    #annotate entity
                    augmentedSentence = annotateEntity(augmentedSentence, entityStartIndex, entityType, len(tokenizedNewEntity),logfilename, debug=debug)
                    #copy labels for tokens after the entity
                    augmentedSentence = annotateAfterEntity(sentence, augmentedSentence, entityEndIndexInOriginal, offset,logfilename, debug=debug)
                    #shift the old entity positions by the offset generated through replacement to find them for further replacements
                    for entity in entitiesInSentence:
                        entity[2] += offset
                    
                    replacements.append((tokenToReplace, candidate))
                    successfulReplacements += 1
                else:
                    failedReplacements += 1
                    log("Something went wrong, found no replacement for token: " + tokenToReplace, logfilename)

            log("\nreplacements (old, new):\n"   + str(replacements), logfilename)
            log("\naugmented sentence: \n" + augmentedSentence.to_tagged_string(), logfilename)

            #add augmented sentence
            keys = [0.01, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
            percent = i/len(corpus)
            for key in keys:
                if percent <= key:
                    if originalSentence.to_tagged_string() != augmentedSentence.to_tagged_string(): #so something in the sentence changed
                        augmentedCorpus[str(key)].append(augmentedSentence)
            
            if originalSentence.to_tagged_string() == augmentedSentence.to_tagged_string():
                log("\nThis sentence did not change during augmentation.", logfilename)
                noChangeAugmentations += 1

        except KeyError:
            log("KeyError - probably encountered out-of-vocabulary word - augmentation of the following sentence failed:\n" + originalSentence.to_plain_string(), logfilename)
        except Exception as e:
            log("Unknown exception occurred - augmentation of the following sentence failed:\n" + originalSentence.to_plain_string(), logfilename)
            traceback.print_exc()

    if save:
        keys = [0.01, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
        for key in keys:
            combineCorpora(originalCorpus[str(key)], augmentedCorpus[str(key)],dataset.dev, dataset.test, dirname + "cMR"+str(key), timestamp=timestamp)
            log("Saved augmented corpus to: ./" + dirname + "cMR"+str(key), logfilename)
            saveCorpus(augmentedCorpus[str(key)], dataset.dev, dataset.test, dirname + "sMR"+str(key),  timestamp=timestamp)
            log("Saved augmented corpus to: ./" + dirname + "sMR"+str(key), logfilename)
            

    time_total = (time.time() - start_time)
    time_sentence = (time.time() - start_time)/len(corpus)
    log("\n" + "%%%%%%%"*100 + "\n\nSTATISTICS:\naugmented " + str(len(augmentedCorpus[str(1.0)])) + "/" +  str(len(corpus)) + " instances", logfilename)
    log("\nno change after aug.:    "+ str(noChangeAugmentations), logfilename)
    log("sentences without entities: " + str(noEntitySentence), logfilename)
    log("irreprod. tokenization: " + str(irreproducibleTokenization), logfilename)
    #log("attempted replacements:  "  + str(successfulReplacements + failedReplacements) + " of " +  str(totalTokens) + " ( " + str((successfulReplacements+failedReplacements)/totalTokens) + ")", logfilename)
    #log("successful replacements: "  + str(successfulReplacements) + " of " +  str(totalTokens) + " ( " + str(successfulReplacements/totalTokens) + ")", logfilename)
    log("total entities:     "       + str(totalEntities), logfilename)
    log("total tokens:       "       + str(totalTokens), logfilename)
    log("--- %s seconds ---" % time_total, logfilename)
    log("--- %s seconds per sentence ---" % time_sentence, logfilename)
    return augmentedCorpus

path = str(sys.argv[1])
dataset = (loadCorpus(path))
dictionary = buildEntityDict(dataset.train)
MentionReplace(dataset, dirname="datasets/", logfilename="log", debug=False, save=True, timestamp=False, entityDictionary=dictionary )
