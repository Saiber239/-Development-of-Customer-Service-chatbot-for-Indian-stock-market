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
import requests

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
import yfinance as yf
import nltk
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')

model.load('./model.tflearn')
words = pickle.load(open('texts1.pkl','rb'))
stemmer = LancasterStemmer()

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# Define a function to preprocess the user's message
def bow(sentence, words, show_details=False):
    # Tokenize the user's message
    sentence_words = clean_up_sentence(sentence)
    # Bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

# Define a function to clean up the user's message
def clean_up_sentence(sentence):
    # Tokenize the user's message
    sentence_words = nltk.word_tokenize(sentence)
    # Remove punctuation and convert to lowercase
    sentence_words = [word.lower() for word in sentence_words if word.isalnum()]
    return sentence_words

# Define a function to predict the intent of the user's message
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.5
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

import re
import random
import yfinance as yf
import json

# Load intents from JSON file
with open('intents.json', encoding='utf-8') as file:
    intents = json.load(file)

def get_stock_response(msg):
    # Check if message contains a valid stock ticker symbol
    pattern = r'^[A-Za-z]{1,5}(\.[A-Za-z]{1,6})?(\.[A-Za-z]{1,4})?$'
    if not re.match(pattern, msg):
        # Check if message contains a valid ticker symbol with exchange name
        pattern = r'^[A-Z]{1,20}\.[A-Z]{2,3}$'
        if not re.match(pattern, msg):
            return None
    
    # Fetch the stock data using the ticker symbol
    print(f"Fetching stock data for {msg}")
    stock_data = yf.download(msg, period='1d')

    if len(stock_data) > 0:
        open_price = stock_data.iloc[-1]['Open']
        close_price = stock_data.iloc[-1]['Close']
        high_price = stock_data.iloc[-1]['High']
        low_price = stock_data.iloc[-1]['Low']
        volume = stock_data.iloc[-1]['Volume']
        result = f"The current price of {msg} is {close_price:.2f}. "\
                 f"The open price was {open_price:.2f}, the high price was {high_price:.2f}, "\
                 f"the low price was {low_price:.2f}, and the volume was {volume:.0f}."
        print(result)
    else:
        result = f"Sorry, I couldn't find any information for {msg}. Please enter a valid stock ticker symbol."
        print(result)

    return result

def getResponse(ints, intents_json):
    if not ints or not isinstance(ints[0], dict) or 'intent' not in ints[0]:
        return "I'm sorry, I'm not sure what you're asking for. Can you please try again?"
    
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            # Find URLs in the response and replace them with hyperlinks
            result = re.sub(r'(https?://\S+)', r'<a href="\1">\1</a>', result)
            break
    return result

def chatbot_response(msg):
    # Check if message is a valid stock ticker symbol
    stock_response = get_stock_response(msg)
    if stock_response:
        return stock_response
    
    # If not, proceed with intent classification
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
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
    response = chatbot_response(userText) # pass the intents dictionary as an argument
    return response

def chatbot_response(msg):
    # Check if message is a valid stock ticker symbol
    stock_response = get_stock_response(msg)
    if stock_response:
        return stock_response
    
    # If not, proceed with intent classification
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port='5000')
