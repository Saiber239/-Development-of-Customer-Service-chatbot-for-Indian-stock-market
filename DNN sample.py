import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import tensorflow as tf
import numpy as np
import tflearn
import random
import json
import pickle
import DNN
import re

words = []
classes = []
documents = []
ignore = ['?']
data_file = open('intents.json', encoding="utf8").read()
intents = json.loads(data_file)

for intent in intents['intents']:
  for pattern in intent['patterns']:
    w = nltk.word_tokenize(pattern)
    words.extend(w)
    documents.append((w, intent['tag']))
    if intent['tag'] not in classes:
      classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)

pickle.dump(words,open('texts1.pkl','wb'))
pickle.dump(classes,open('labels1.pkl','wb'))

training = []
output = []

output_empty = [0]*len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])
print("Training data Created")

tf.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, 10)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.fit(train_x, train_y, n_epoch=100, batch_size=8, show_metric=True)
model.save('model.tflearn')
print("model created")


nltk.download('popular')

import pickle
import numpy as np

model.load('./model.tflearn')
import json
import random
intents = json.loads(open('intents.json', encoding="utf8").read())
words = pickle.load(open('texts1.pkl','rb'))
classes = pickle.load(open('labels1.pkl','rb'))

def clean_up_sentence(sentence):
  sentence_words = nltk.word_tokenize(sentence)
  sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
  return sentence_words

def bow(sentence, words, show_details=False):
  sentence_words = clean_up_sentence(sentence)
  bag = [0]*len(words)
  for s in sentence_words:
    for i,w in enumerate(words):
      if w == s:
        bag[i] = 1
        if show_details:
          print("found in bag: %s" % w)
  
  return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        if classes[r[0]] == 'stock':
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        else:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

import yfinance as yf

def get_ticker_symbol(company_name):
    try:
        ticker = yf.Ticker(company_name)
        symbol = ticker.info.get('symbol')
        if symbol is None:
            return None
        else:
            return symbol
    except:
        return None
    
def getResponse(ints, sentence):
    tag = ints[0]['intent']
    stock_name = None
    
    # Look for the stock name in the user input
    for word in re.findall(r'\b[A-Za-z0-9]+\.[A-Za-z0-9]+\b|\b[A-Za-z0-9]+\b', sentence):
        if get_ticker_symbol(word) is not None:
            stock_name = word
            break
    
    if stock_name is None:
        return "Sorry, I couldn't find a valid stock name in your input."
    
    # Get the stock ticker symbol using the company name
    stock_ticker = get_ticker_symbol(stock_name)
    
    if stock_ticker:
        # Show the ticker symbol to the user
        response = f"The ticker symbol is {stock_ticker}."
    else:
        response = f"Sorry, I couldn't find any information for {stock_name}. Please enter a valid company name or ticker symbol."
    
    # Fetch the stock data using the ticker symbol
    stock_data = yf.download(stock_ticker, period='1d')
    if len(stock_data) > 0:
        open_price = stock_data.iloc[-1]['Open']
        close_price = stock_data.iloc[-1]['Close']
        high_price = stock_data.iloc[-1]['High']
        low_price = stock_data.iloc[-1]['Low']
        volume = stock_data.iloc[-1]['Volume']
        result = f"The current price of {stock_name} is {close_price:.2f}. "\
                 f"The open price was {open_price:.2f}, the high price was {high_price:.2f}, "\
                 f"the low price was {low_price:.2f}, and the volume was {volume:.0f}."
    else:
        result = f"Sorry, I couldn't find any information for {stock_name}. Please enter a valid company name or ticker symbol."
    
    return response + ' ' + result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, msg)
    return res

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, msg)
    return res

if __name__ == "__main__":
    app.run(debug = False, host = '0.0.0.0', port = '5000')
    
