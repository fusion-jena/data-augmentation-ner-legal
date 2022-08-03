from utils import *
import sys

# use like this:
# python3 aug/synonym_replace.py path percentage source

def SynonymReplace(dataset, p, modelname, model, dirname, logfilename="log", debug=False, save=False, timestamp=True, deterministic=True):
    corpus = dataset.train
    #mysql database info for thesaurus replacement
    HOST = "ENTER IP HERE"
    USER = "ENTER DATABASE USERNAME HERE"
    PASSWORD = "ENTER PASSWORD HERE" #! fill in before using thesaurus as replacement source
    DATABASE = "ENTER DATABASE NAME HERE"

    start_time = time.time()
    logfilename = logfilename +str(p)+"p" + modelname + "_SR"
    
    originalCorpus = {"0.01" : [], "0.1": [], "0.3":[], "0.5":[], "0.7":[], "0.8":[], "0.9":[], "1.0":[]}
    augmentedCorpus = {"0.01" : [], "0.1": [], "0.3":[], "0.5":[], "0.7":[], "0.8":[], "0.9":[], "1.0":[]}

    successfulReplacements = failedReplacements = noChangeAugmentations = totalTokens = irreproducibleTokenization = totalEntities = 0
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%Z")

    log("SynonymReplacement - " +current_time +  "\n\nCONFIGURATION:",logfilename)
    log("corpus-size:         " + str(len(corpus)), logfilename)
    log("corpus-hash:         " + str(hash(corpus)), logfilename)
    log("modelname:           " + str(modelname),logfilename)
    log("modify entities:     " + str(False), logfilename)
    log("modify non-entities: " + str(True),logfilename)
    log("modify-percentage:   " + str(p),logfilename)
    log("debug:               " + str(debug),logfilename)
    log("save:                " + str(save),logfilename)


    if modelname == "thesaurus":
        connectedToDatabase = False
        try:
            cnx = mysql.connector.connect(
                host=HOST,
                user=USER,
                password=PASSWORD,
                database=DATABASE
            )
            model =  cnx.cursor()
            connectedToDatabase = True
        except:
            log("Warning: Unable to connect to database. Using public API." + "", logfilename)
            model = ""
            

    for i in tqdm(range(len(corpus))):
        sentence = corpus[i]
        originalSentence = augmentedSentence = sentence
        #augmentedSentence = sentence
        totalTokens += len(sentence)

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

            replacements = []
            #get indices of non-entity and entity tokens
            _, otherIndices = getIndicesByClass(augmentedSentence, logfilename) 

            if debug:
                log("non-entity indices: " + str(otherIndices), logfilename)

            #shuffle the order of non-entity tokens
            random.shuffle(otherIndices) 
            targetModificationsCount = floor(p*len(otherIndices)) #this is relative to the number of tokens that match the regex, not the total number

            # loop through list of non-entity tokens and replace until target percentage is reached
            for j in range(0, targetModificationsCount):
                #set sentence to the augmented sentence from last iteration, has labels, no change yet
                sentence = augmentedSentence

                tokenStartIndex = otherIndices[j]
                tokenToReplace = sentence[tokenStartIndex].text
                tokenizedOriginalToken = tokenize(tokenToReplace)

                #returns -1 if no replacement available
                candidate = getReplacement(sentence, sentence.to_plain_string(), tokenToReplace, tokenStartIndex, modelname, model, logfilename,deterministic=deterministic, debug=debug)

                if candidate != -1:      
                    #tokenize the new token
                    tokenizedNewToken = tokenize(candidate)
                    #get entity end index in the old sentence
                    tokenEndIndexInOriginal = tokenStartIndex + len(tokenizedOriginalToken) #this is not required, should always equal 1
                    #get offset for labels of new and old sentence
                    offset = len(tokenizedNewToken) - len(tokenizedOriginalToken) #is always 0, since we replace one token with another token (regex assures replacement is only one token)
                    assert offset == 0
                    #replace old entity with new entity, return sentence without any labels
                    augmentedSentence = replaceSpan(sentence, tokenizedOriginalToken, tokenizedNewToken, tokenStartIndex,logfilename, debug=debug )
                    #copy labels from original sentence until entity
                    augmentedSentence = annotateBeforeEntity(sentence, augmentedSentence, tokenStartIndex,logfilename, debug=debug)
                    #annotate entity
                    augmentedSentence = annotateNonEntity(augmentedSentence, tokenStartIndex, len(tokenizedNewToken),logfilename, debug=debug)
                    #copy labels for tokens after the entity
                    augmentedSentence = annotateAfterEntity(sentence, augmentedSentence, tokenEndIndexInOriginal, offset,logfilename, debug=debug)
                    #shift the old positions by the offset generated through replacement to find them for further replacements
                    
                    replacements.append((tokenToReplace, candidate))
                    successfulReplacements += 1
                else:
                    failedReplacements += 1
                    log("Something went wrong, no change to token: " + tokenToReplace, logfilename)

            log("\nreplacements (old, new):\n"   + str(replacements), logfilename)
            log("\ntagged augmented sentence:\n" + augmentedSentence.to_tagged_string(), logfilename) 


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


        except AssertionError:
            log("AssertionError - Offset did not equal 0, sentence index:" + str(i), logfilename)
            traceback.print_exc()
        except KeyError:
            log("KeyError - probably encountered out-of-vocabulary word - augmentation of the following sentence failed:\n" + originalSentence.to_plain_string(), logfilename)
            traceback.print_exc()
        except Exception as e:
            log("Unknown exception occurred - augmentation of the following sentence failed:\n" + originalSentence.to_plain_string(), logfilename)
            traceback.print_exc()

    if modelname == "fasttext":
        modelname = "ftx"
    elif modelname == "thesaurus":
        modelname = "the"
    elif modelname == "clm":
        modelname = "clm"

    if save:
        keys = [0.01, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
        for key in keys:
            combineCorpora(originalCorpus[str(key)], augmentedCorpus[str(key)], dataset.dev, dataset.test, dirname + "cSR"+str(key)+"f"+str(p)+"p" + modelname, timestamp=timestamp)
            log("Saved augmented corpus to: ./" + dirname + "cSR"+str(key)+"f"+str(p)+"p" + modelname, logfilename)
            saveCorpus(augmentedCorpus[str(key)],dataset.dev, dataset.test, dirname + "sSR"+str(key)+"f"+str(p)+"p" + modelname,  timestamp=timestamp)
            log("Saved augmented corpus to: ./" + dirname + "sSR"+str(key)+"f"+str(p)+"p" + modelname, logfilename)

    time_total = (time.time() - start_time)
    time_sentence = (time.time() - start_time)/len(corpus)
    log("\n" + "%%%%%%%"*100 + "\n\nSTATISTICS:\naugmented " + str(len(augmentedCorpus[str(1.0)])) + "/" +  str(len(corpus)) + " instances", logfilename)
    log("\nno change after aug.:    "+ str(noChangeAugmentations), logfilename)
    log("attempted replacements:  "  + str(successfulReplacements + failedReplacements) + " of " +  str(totalTokens) + " ( " + str((successfulReplacements+failedReplacements)/totalTokens) + ")", logfilename)
    log("successful replacements: "  + str(successfulReplacements) + " of " +  str(totalTokens) + " ( " + str(successfulReplacements/totalTokens) + ")", logfilename)
    log("irreprod. tokenization: " + str(irreproducibleTokenization), logfilename)
    log("total non-entities: "       + str(totalTokens-totalEntities), logfilename)
    log("total tokens:       "       + str(totalTokens), logfilename)
    log("--- %s seconds ---" % time_total, logfilename)
    log("--- %s seconds per sentence ---" % time_sentence, logfilename)
    return augmentedCorpus

path = str(sys.argv[1])
percentage = float(sys.argv[2]) #0.2, 0.4, 0.6 (but any is possible) 
source = str(sys.argv[3]) #thesaurus, fasttext, clm (glove, word2vec)

dataset = (loadCorpus(path))
print("initializing source...")
sourceModel = initSource(source)
print("starting augmentation...")
SynonymReplace(dataset, percentage, source, sourceModel, dirname="datasets/",logfilename="log_", debug=False, save=True, timestamp=False)