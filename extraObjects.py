import tensorflow as tf
import keras
import statistics
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer 







#Function to preprocess texts for single prediction
def preprocessComment(comment):
  MAX_SEQUENCE_LENGTH = 2000
  tokenizer = RegexpTokenizer(r'[A-Za-z]+')
  tokenized_text = tokenizer.tokenize(comment)

  stemmer = SnowballStemmer("english")
  stemmed_words = [stemmer.stem(word) for word in tokenized_text]

  stemmed_sentence = ' '.join(stemmed_words)

  tok = Tokenizer(oov_token='<UNK>')
  # fit the tokenizer on the documents
  tok.fit_on_texts([stemmed_sentence])
  tok.word_index['<PAD>'] = 0
  max([(k, v) for k, v in tok.word_index.items()], key = lambda x:x[1]), min([(k, v) for k, v in tok.word_index.items()], key = lambda x:x[1]), tok.word_index['<UNK>']
  text_sequence = tok.texts_to_sequences([stemmed_sentence])
  print(len(text_sequence[0]))


  input_data = sequence.pad_sequences(text_sequence, maxlen=MAX_SEQUENCE_LENGTH)
  return input_data


def ensemblePrediction(predictionList):
    predictionList.sort()
    indecisiveVotes = [[0,0,1,1], [0,0,2,2], [1,1,2,2]]
    if predictionList in indecisiveVotes:
        ensembleDecision = 2

    else:
        ensembleDecision = statistics.mode(predictionList)

    return ensembleDecision


#Function to preprocess texts for batch prediction
def preprocessBatch(comments):
  MAX_SEQUENCE_LENGTH = 2000
  tokenizer = RegexpTokenizer(r'[A-Za-z]+')
  stemmed_sentences = []
  for comment in comments:
    tokenized_text = tokenizer.tokenize(comment)

    stemmer = SnowballStemmer("english")
    stemmed_words = [stemmer.stem(word) for word in tokenized_text]

    stemmed_sentence = ' '.join(stemmed_words)
    stemmed_sentences.append(stemmed_sentence)

  tok = Tokenizer(oov_token='<UNK>')
  # fit the tokenizer on the documents
  tok.fit_on_texts(stemmed_sentences)
  tok.word_index['<PAD>'] = 0
  max([(k, v) for k, v in tok.word_index.items()], key = lambda x:x[1]), min([(k, v) for k, v in tok.word_index.items()], key = lambda x:x[1]), tok.word_index['<UNK>']
  
  text_sequence = tok.texts_to_sequences(stemmed_sentences)
  #print(len(text_sequence[0]))


  input_data = sequence.pad_sequences(text_sequence, maxlen=MAX_SEQUENCE_LENGTH)
  return input_data