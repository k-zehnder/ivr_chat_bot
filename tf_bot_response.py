import json
# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle



class ModelConfig:
    def __init__(self, training_data, intents_json) -> None:
        self.training_data = training_data
        self.intents_json = intents_json
        self.data = self.restore_dsa()
        self.model = self.load_model()
        
    def load_model(self):
        net = self._build_net()
        return tflearn.DNN(net, tensorboard_dir="tflearn")
    
    def _build_net(self):
        # Build neural network
        net = tflearn.input_data(shape=[None, len(self.data["train_x"][0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(self.data["train_y"][0]), activation='softmax')
        net = tflearn.regression(net)
        return net
    
    def load_intents(self):
        with open(self.intents_json) as json_data:
            intents = json.load(json_data)
        return intents
    
    def restore_dsa(self):
        data = pickle.load(open(self.training_data, "rb" ))
        self.words = data['words']
        self.classes = data['classes']
        self.train_x = data['train_x']
        self.train_y = data['train_y']
        return {
            "words" : self.words,
            "classes" : self.classes,   
            "train_x" : self.train_x,
            "train_y" : self.train_y
        }
        
    def __str__(self):
        return str(self.data["classes"])

class BotResponse(ModelConfig):
    def __init__(self, training_data, intents_json) -> None:
        super().__init__(training_data, intents_json)
        # create a data structure to hold user context
        self.context = {}
        self.ERROR_THRESHOLD = 0.25
    
    def clean_up_sentence(self, sentence):
        # tokenize the pattern
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def bow(self, sentence, show_details=False):
        # tokenize the pattern
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words
        bag = [0]*len(self.data["words"])  
        for s in sentence_words:
            for i,w in enumerate(self.data["words"]):
                if w == s: 
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)

        return(np.array(bag))

    def classify(self, sentence):
        # generate probabilities from the model
        # results = self.model.predict([self.bow(sentence, self.data["words"])])[0]
        model = self.model
        bow = self.bow
        words = self.data["words"]
        results = model.predict([bow(sentence, words)])[0]
        print(results)

        classes = self.data["classes"]
        # filter out predictions below a threshold
        results = [[i,r] for i,r in enumerate(results) if r>self.ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((classes[r[0]], r[1]))
        # return tuple of intent and probability
        return return_list

    def response(self, sentence, userID='123', show_details=False):
        intents = self.load_intents()
        results = self.classify(sentence)
        # if we have a classification then find the matching intent tag
        if results:
            # loop as long as there are matches to process
            while results:
                for i in intents['intents']:
                    # find a tag matching the first result
                    if i['tag'] == results[0][0]:
                        # set context for this intent if necessary
                        if 'context_set' in i:
                            if show_details: 
                                print ('context:', i['context_set'])
                            self.context[userID] = i['context_set']

                        # check if this intent is contextual and applies to this user's conversation
                        if not 'context_filter' in i or \
                            (userID in self.context and 'context_filter' in i and i['context_filter'] == self.context[userID]):
                            if show_details: 
                                print ('tag:', i['tag'])
                            # a random response from the intent
                            return print(random.choice(i['responses']))

                results.pop(0)

bot = BotResponse('training_data', '/home/batman/Desktop/py/ivr_chat_bot/intents.json')
print(bot)
print(bot.bow("log my blood pressure"))
print(bot.classify("log my blood pressure"))
print(bot.response("log my blood pressure"))

# # p = bow("is your shop open today?", words)
# # print (p)
# # print (classes)
# print("\n" + "*"*14)
# user_question = "log my blood pressure"        
# print(classify(user_question))
# # print(response(user_question, show_details=True))
# print(f'context: {context}')