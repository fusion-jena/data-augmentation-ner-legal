from utils import *
import sys
from time import sleep
import pickle
import shutil

def logCorpusReport(originalCorpus, augmentedCorpus, logfilename):
    log("Augmented Corpus Size(s):" + " ", logfilename)
    for kex in augmentedCorpus.keys():
        log("Fraction: " + str(kex) + " Size: " + str(len(augmentedCorpus[str(kex)])), logfilename)
    log("Original Corpus Size(s):" + " ", logfilename)
    for key in originalCorpus.keys():
        log("Fraction: " + str(key) + " Size: " + str(len(originalCorpus[str(key)])), logfilename)

def connect():
    trans = BackTranslation(url=[
        'translate.google.com',
        'translate.google.co.kr',
        ], proxies={'http': '127.0.0.1:1234', 'http://host.name': '127.0.0.1:4012'})
    return trans

def backtranslateGoogleOld(connection, text, source_language_code, intermed_language_code):
    return connection.translate(text, src=source_language_code, tmp = intermed_language_code, sleeping = 2.5).result_text

def saveCheckpoint(statistics, originalCorpus, augmentedCorpus, corpus, logfilename):
    if os.path.exists("./bt_checkpoints/"):
        shutil.rmtree("./bt_checkpoints/")

    os.makedirs("./bt_checkpoints/")
    os.makedirs("./bt_checkpoints/org/")
    os.makedirs("./bt_checkpoints/aug/")
    
    with open('./bt_checkpoints/checkpoint.pkl', 'wb') as file:
        pickle.dump(statistics, file)  

    for key in originalCorpus.keys():
        with open("./bt_checkpoints/org/" + str(key) + ".txt", "w") as myfile:
            for s in originalCorpus[str(key)]:
                for token in s:
                    myfile.write(token.text + " " + token.get_tag('ner').value + "\n")
                myfile.write("\n")

    for kex in augmentedCorpus.keys():
        with open("./bt_checkpoints/aug/" + str(kex) + ".txt", "w") as thefile:
            for s in augmentedCorpus[str(kex)]:
                for token in s:
                    thefile.write(token.text + " " + token.get_tag('ner').value + "\n")
                thefile.write("\n")

    Path('./bt_checkpoints/dummytest.txt').touch()
    Path('./bt_checkpoints/dummydev.txt').touch()
    Path('./bt_checkpoints/' + str(statistics[0]/len(corpus) * 100) + '.txt').touch()
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%Z")
    log( "\n"+"$&$&$"*100 + "\n" + "\n CREATED CHECKPOINT: sentence " + str(statistics[0])+  " - " +  str(statistics[0]/len(corpus) * 100) + "% - " + current_time + ")", logfilename)
    logCorpusReport(originalCorpus, augmentedCorpus, logfilename)
    log("$&$&$"*100 + "\n", logfilename)


def loadCheckpoint(corpus, logfilename):
    
    originalCorpus = {"0.01" : [], "0.1": [], "0.3":[], "0.5":[], "0.7":[],"0.8":[], "0.9":[], "1.0":[]}
    augmentedCorpus = {"0.01" : [], "0.1": [], "0.3":[], "0.5":[], "0.7":[],"0.8":[], "0.9":[], "1.0":[]}

    with open('./bt_checkpoints/checkpoint.pkl', 'rb') as myfile:
        statistics = pickle.load(myfile)
        i = statistics[0]
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%Z")
        log( "\n"+"$&$&$"*100 + "\n" + "\n RESTORING CHECKOPOINT: sentence " + str(i)+  " - " +  str(i/len(corpus) * 100) + "% - " + current_time + ")\n" + "$&$&$"*100 , logfilename)

        columns = {0: 'text', 1: 'ner'}

        if os.path.exists("./bt_checkpoints/last/crashIndex.txt"):
            with open('./bt_checkpoints/last/crashIndex.txt') as f:
                crashIndex = int(f.read())
        
        for key in originalCorpus.keys():
            data =  ColumnCorpus("./bt_checkpoints/org/", columns,
                train_file=str(key)+ ".txt",
                test_file='../dummytest.txt',
                dev_file='../dummydev.txt')
            for entry in data.train:
                originalCorpus[str(key)].append(entry)

        for kex in augmentedCorpus.keys():
            data =  ColumnCorpus("./bt_checkpoints/aug/", columns,
                train_file=""+str(kex)+ ".txt",
                test_file='../dummytest.txt',
                dev_file='../dummydev.txt')
            for entry in data.train:
                augmentedCorpus[str(kex)].append(entry)

    logCorpusReport(originalCorpus, augmentedCorpus, logfilename)

    return statistics, crashIndex, originalCorpus, augmentedCorpus

def backtranslateAugment(dataset, engine="google-free", dirname="backtranslated", logfilename="log", debug=True, save=False, timestamp=True):
    corpus = dataset.train
    start_time = time.time()
    logfilename = logfilename + "_BT"
    successWithEntities = successWithoutEntities = noChange = foundOriginal = foundBacktranslated = irreproducibleTokenization = 0

    connection = connect()

    originalCorpus = {"0.01" : [], "0.1": [], "0.3":[], "0.5":[], "0.7":[],"0.8":[], "0.9":[], "1.0":[]}
    augmentedCorpus = {"0.01" : [], "0.1": [], "0.3":[], "0.5":[], "0.7":[],"0.8":[], "0.9":[], "1.0":[]}

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%Z")

    log("BackTranslation - " +current_time +  "\n\nCONFIGURATION:",logfilename)
    log("corpus-size:         " + str(len(corpus)), logfilename)
    log("corpus-hash:         " + str(hash(corpus)), logfilename)
    log("engine:              " + str(engine),logfilename)
    log("debug:               " + str(debug),logfilename)
    log("save:                " + str(save),logfilename)

    checkpointAgent = 0
    crashIndex = -2

    if os.path.exists('./bt_checkpoints/checkpoint.pkl'):
        statistics, crashIndex, originalCorpus, augmentedCorpus = loadCheckpoint(corpus, logfilename)
        i, successWithEntities, successWithoutEntities, noChange, foundOriginal, foundBacktranslated, irreproducibleTokenization, time_total = statistics
        start = i

    else:
        start = 0

    for i in range(start, len(corpus)):
        sentence = corpus[i]
        originalSentence = sentence
        checkpointAgent += 1

        if i == crashIndex:
            log("Warning: Skipping sentence with index " + str(crashIndex) + " as it caused the last crash...", logfilename)
            continue

        if os.path.exists("./bt_checkpoints/last"):
            shutil.rmtree("./bt_checkpoints/last")

        os.makedirs("./bt_checkpoints/last/")
        with open("./bt_checkpoints/last/crashIndex.txt", "w") as myfile:
            myfile.write(str(i))

        if checkpointAgent > 0.05 * len(corpus) or i == (crashIndex + 1):
            time_total = (time.time() - start_time)
            statistics = (i, successWithEntities, successWithoutEntities, noChange, foundOriginal, foundBacktranslated, irreproducibleTokenization, time_total)  
            saveCheckpoint(statistics, originalCorpus, augmentedCorpus, corpus, logfilename)
            checkpointAgent = 0
            
        #add original sentence
        keys = [0.01, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
        percent = float(i)/float(len(corpus))
        for key in keys:
            if percent <= float(key):
                originalCorpus[str(key)].append(originalSentence)

        #sanity check - if we can reproduce the way it is tokenized, important for further handling of the sentence
        if (len(sentence) != len(tokenize(sentence.to_plain_string()))):
            irreproducibleTokenization += 1
            continue

        #extract entities from original sentence
        entities = getEntities(sentence) #returns tokenToReplace, entityType, entityStartIndex
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f%Z")
        log( "\n"+"%%%%%%%%%"*100 + "\n" + "\ntagged original sentence: (" + str(i)+  " - " +  str(i/len(corpus) * 100) + "% - " + current_time + ")\n"  + originalSentence.to_tagged_string(), logfilename)
        log("\noriginal entities:\n" + str(entities), logfilename)

        if ContainsSameSequenceDifferentLabel(entities, sentence):
            log("\n\Skipping due to ContainsSameSequenceDifferentLabel: Sentence index : " + str(i) + "", logfilename)
            log(sentence.to_tagged_string() + "\n", logfilename)
            continue

        #backtranslate orignal sentence
        if engine == "google":
            result = backtranslateGoogle(sentence.to_original_text(),"de", "en")
        elif engine == "google-free":
            result = backtranslateGoogleOld(connection, sentence.to_original_text(),"de", "en")
        elif engine == "deepl":
            result = backtranslateDeepL(sentence.to_original_text())
        else:
            raise Exception("select valid translation engine!")

        currentEntity = lastEntityTag = ""
        #tokenize backtranslated sentence
        newSentence = tokenize(result) 

        try:
            #backtranslate the original entities (and remove appended dot)
            backtranslatedEntities = []
            for entity in entities:
                originalEntity = entity[0]
                time.sleep(1) # neccessary to prevent API from blocking the requests

                if engine == "google":
                    backtranslatedEntity = backtranslateGoogle(originalEntity,"de", "en")
                elif engine == "google-free":
                    backtranslatedEntity = backtranslateGoogleOld(connection, originalEntity,"de", "en")
                elif engine == "deepl":
                    backtranslatedEntity = backtranslateDeepL(originalEntity)
                else:
                    raise Exception("select valid translation engine!")

                #remove appended dot
                if "." == backtranslatedEntity[-1] and originalEntity[-1] != ".":
                    p = backtranslatedEntity.split(".")
                    if debug:
                        log("removed dot: " + backtranslatedEntity + " -> " + p[0], logfilename)
                    backtranslatedEntity = p[0]
                entity.append(backtranslatedEntity)

            log("translated entities:\n" + str(entities), logfilename)

            #annotate the backtranslated sentence
            newSentence, foundOriginalTmp, foundBacktranslatedTmp = annotateEntities(newSentence, entities, debug, logfilename)
            foundOriginal += foundOriginalTmp
            foundBacktranslated += foundBacktranslatedTmp


            #annotate the non-entities
            for xi in range(0, len(newSentence)):
                if newSentence[xi].get_tag('ner').value == "":
                    newSentence[xi].add_tag('ner', "O")

            log("\ntagged augmented sentence:\n" + newSentence.to_tagged_string(), logfilename) 

            #add augmented sentence
            keys = [0.01, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
            percent = float(i)/float(len(corpus))
            for key in keys:
                if percent <= float(key):
                    if sentence.to_tagged_string() != newSentence.to_tagged_string(): #so something in the sentence changed
                        augmentedCorpus[str(key)].append(newSentence)
                        
            #just relevant for statistics
            if (len(entities) == 0):
                successWithoutEntities += 1
            elif sentence.to_tagged_string() == newSentence.to_tagged_string() :
                log("\nThis sentence did not change during augmentation.", logfilename)
                noChange += 1
            else:
                successWithEntities += 1

        except KeyError as e:
            log(str(e), logfilename)
            log("\nAugmentation failed",logfilename)
            log("Back-Translated:\n" + newSentence.to_plain_string(),logfilename)
        except Exception as e:
            log("\nUnknown exception occurred - augmentation of the following sentence failed:\n" + originalSentence.to_plain_string(), logfilename)
            traceback.print_exc()
        
    if save:
        keys = [0.01, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
        for key in keys:
            combineCorpora(originalCorpus[str(key)], augmentedCorpus[str(key)],dataset.dev, dataset.test, dirname + "cBT"+str(key), timestamp=timestamp)
            log("\nSaved augmented corpus to: ./" + dirname + "cBT"+str(key), logfilename)
            saveCorpus(augmentedCorpus[str(key)],dataset.dev, dataset.test, dirname + "sBT"+str(key), timestamp=timestamp)
            log("\nSaved augmented corpus to: ./" + dirname + "sBT"+str(key), logfilename)

    time_total = (time.time() - start_time)
    time_sentence = (time.time() - start_time)/len(corpus)
    log("\n" + "%%%%%%%"*100 + "\n\nSTATISTICS:\naugmented " + str(len(augmentedCorpus[str(1.0)])) + "/" +  str(len(corpus)) + " instances", logfilename)
    log("\nno change after aug.:    "+ str(noChange), logfilename)
    log("irreprod. tokenization: " + str(irreproducibleTokenization), logfilename)
    log("augmented sentences not containing entities: " + str(successWithoutEntities), logfilename)
    log("augmented sentences containing entities:    " + str(successWithEntities), logfilename)
    log("found original entity:                      " + str(foundOriginal), logfilename)
    log("found backtranslated entity:                " + str(foundBacktranslated), logfilename)
    log("--- %s seconds ---" % time_total, logfilename)
    log("--- %s seconds per sentence ---" % time_sentence, logfilename)
    return augmentedCorpus

path = str(sys.argv[1])
dataset = (loadCorpus(path))
backtranslateAugment(dataset, dirname="datasets/", logfilename="log", debug=False, save=True, timestamp=False)