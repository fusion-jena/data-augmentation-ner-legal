# -*- coding: utf-8 -*-
import sys
import flair
import time
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from aug import utils

start_time = time.time()
path = str(sys.argv[1])
pathTest = str(sys.argv[2])
learningRate = float(sys.argv[3])
batchSize = int(sys.argv[4])
#TrainWithDev = str(sys.argv[5])
TrainWithDev = False
storage = str(sys.argv[5])

if TrainWithDev == "True":
    TrainWithDev = True
else:
    TrainWithDev = False

dataset = (utils.loadCorpus(path))
ner_dictionary = dataset.make_label_dictionary(label_type='ner')

# 4. initialize embedding stack with Flair and GloVe
embedding_types = [
    WordEmbeddings('de'),
    FlairEmbeddings('de-forward'),
    FlairEmbeddings('de-backward'),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
tagger = SequenceTagger(hidden_size=256,
                        dropout=0.25,
                        embeddings=embeddings,
                        tag_dictionary=ner_dictionary,
                        tag_type="ner",
                        use_rnn=True,
                        use_crf=True)

# 6. initialize trainer
trainer = ModelTrainer(tagger, dataset)

# 7. start training
trainer.train('resources/taggers/sota-ner-flair',
              embeddings_storage_mode=storage, #cpu for big datasets (more than 60,000 sentences), gpu otherwise (is faster)
              train_with_dev=TrainWithDev,
              learning_rate=learningRate,    #0.1
              mini_batch_size=batchSize, #32
              patience = 5,   # 5, akbik 2018   
              max_epochs=150) # 150, akbik 2018

"""## Evaluate"""

classifier = SequenceTagger.load('resources/taggers/sota-ner-flair/final-model.pt')

flair.set_seed(123)

if (TrainWithDev):
    datasetTest = (utils.loadCorpus(pathTest))
    results = classifier.evaluate(datasetTest.test, gold_label_type="ner")
else:
    results = classifier.evaluate(dataset.test, gold_label_type="ner")

rep = results.classification_report

#sentences, f1-macro, precision (macro), recall (macro), f1-micro, precision (micro), recall (micro), augmentation method, fraction, replacement percentage, replacement source, training time
time_total = (time.time() - start_time)
csvrow = csvrow = [len(dataset.train),"bilstm", round(rep['macro avg']['f1-score'],4),round(rep['macro avg']['precision'],4), round(rep['macro avg']['recall'],4), round(rep['micro avg']['f1-score'],4), round(rep['micro avg']['precision'],4), round(rep['micro avg']['recall'],4), path[10:12],path[12:15],path[16:19],path[20:23], int(time_total/60)]
csvrow = [str(x) for x in csvrow]
csvrow = ','.join(csvrow)

with open('bilstm_results.csv','a') as fd:
    fd.write(csvrow + "\n")
