# -*- coding: utf-8 -*-
import sys
import os
import flair
import time
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from aug import utils

start_time = time.time()
path = str(sys.argv[1])
pathTest = str(sys.argv[2])
batchSize = int(sys.argv[3])
epochNum = int(sys.argv[4])
#TrainWithDev = str(sys.argv[5])
TrainWithDev = False

if TrainWithDev == "True":
    TrainWithDev = True
else:
    TrainWithDev = False

dataset = (utils.loadCorpus(path))

ner_dictionary = dataset.make_label_dictionary(label_type='ner')

# 2. what label do we want to predict?
label_type = 'ner'

# 4. initialize fine-tuneable transformer embeddings
embeddings = TransformerWordEmbeddings(model='xlm-roberta-large',
                                       layers="-1",                 
                                       subtoken_pooling="first",    
                                       fine_tune=True,
                                       use_context=False,
                                       )

# 5. initialize bare-bones sequence tagger (no CRF, no RNN, no reprojection)
tagger = SequenceTagger(hidden_size=256, #is a required parameter, but doesnt do anything with use_rnn=False
                        embeddings=embeddings,
                        tag_dictionary=ner_dictionary,
                        tag_type='ner',
                        use_crf=False,
                        use_rnn=False,
                        reproject_embeddings=False,
                        )

# 6. initialize trainer
trainer = ModelTrainer(tagger, dataset)

# 7. run fine-tuning
trainer.fine_tune('resources/taggers/sota-ner-flert',
                  learning_rate=5.0e-6,            #following schweter et al
                  mini_batch_size=batchSize,      
                  max_epochs=epochNum,           
                  train_with_dev=TrainWithDev
                  )


"""## Evaluate"""

classifier = SequenceTagger.load('resources/taggers/sota-ner-flert/final-model.pt')

flair.set_seed(123)

if (TrainWithDev):
    datasetTest = (utils.loadCorpus(pathTest))
    results = classifier.evaluate(datasetTest.test, gold_label_type="ner")
else:
    results = classifier.evaluate(dataset.test, gold_label_type="ner")

rep = results.classification_report

#sentences, f1-macro, precision (macro), recall (macro), f1-micro, precision (micro), recall (micro), augmentation method, fraction, replacement percentage, replacement source, training time
time_total = (time.time() - start_time)
csvrow = csvrow = [len(dataset.train),"transformer", round(rep['macro avg']['f1-score'],4),round(rep['macro avg']['precision'],4), round(rep['macro avg']['recall'],4), round(rep['micro avg']['f1-score'],4), round(rep['micro avg']['precision'],4), round(rep['micro avg']['recall'],4), path[10:12],path[12:15],path[16:19],path[20:23], int(time_total/60)]
csvrow = [str(x) for x in csvrow]
csvrow = ','.join(csvrow)

with open('flert_results.csv','a') as fd:
    fd.write(csvrow + "\n")