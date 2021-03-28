import numpy as np 
import string 
import sys 
from nltk.tokenize import RegexpTokenizer 
from nltk.corpus import stopwords 
from keras.models import Sequential
from keras.layers import Dense,Dropout,LSTM
from keras.utils import np_utils 
from keras.callbacks import ModelCheckpoint
import nltk
nltk.download('stopwords')
stop=set(stopwords.words('english'))
punc=list(string.punctuation)
stop.update(punc)
from google.colab import files
text = files.upload()
file_name = "breast_cancer.csv"
text = text[file_name].decode("utf-8")
def tokenize_words(text):
  text = text.lower()
  tokenizer = RegexpTokenizer(r'\w+')
  tokens = tokenizer.tokenize(text)
  filtered = filter(lambda token: token not in stop,tokens)
  return " ".join(filtered)
  
processed_input = tokenize_words(text)
len(processed_input)
char = sorted(list(set(processed_input)))
char_to_num = dict((c,i) for i,c in enumerate(char))
input_len = len(processed_input)
vocab_len = len(char)
print("Total Number of Characters : ",input_len)
print("Total Vocab : ",vocab_len)
seq_len = 100
x_data = []
y_data = []

x = np.reshape(x_data,(n_patterns,seq_len,1))
x = x/float(vocab_len)
y = np_utils.to_categorical(y_data)
model = Sequential()
model.add(LSTM(256,input_shape=(x.shape[1],x.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256,return_sequences=True ))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))
model.compile(loss = 'categorical_crossentropy',optimizer='adam')
model.fit(x,y,epochs=1,batch_size=256,callbacks=desired_callbacks)
