import re
import nltk.corpus
from nltk.corpus import nps_chat
import pandas as pd

class IsQuestion():
    
    # Init constructor
    def __init__(self):
        posts = self.__get_posts()
        feature_set = self.__get_feature_set(posts)
        self.classifier = self.__perform_classification(feature_set)
        
    # Method (Private): __get_posts
    # Input: None
    # Output: Posts (Text) from NLTK's nps_chat package
    def __get_posts(self):
        return nltk.corpus.nps_chat.xml_posts()
    
    # Method (Private): __get_feature_set
    # Input: Posts from NLTK's nps_chat package
    # Processing: 1. preserve alpha numeric characters, whitespace, apostrophe
    # 2. Tokenize sentences using NLTK's word_tokenize
    # 3. Create a dictionary of list of tuples for each post where tuples index 0 is the dictionary of words occuring in the sentence and index 1 is the class as received from nps_chat package 
    # Return: List of tuples for each post
    def __get_feature_set(self, posts):
        feature_list = []
        for post in posts:
            post_text = post.text            
            features = {}
            words = nltk.word_tokenize(post_text)
            for word in words:
                features['contains({})'.format(word.lower())] = True
            feature_list.append((features, post.get('class')))
        return feature_list
    
    # Method (Private): __perform_classification
    # Input: List of tuples for each post
    # Processing: 1. Divide data into 80% training and 10% testing sets
    # 2. Use NLTK's Multinomial Naive Bayes to perform classifcation
    # 3. Print the Accuracy of the model
    # Return: Classifier object
    def __perform_classification(self, feature_set):
        training_size = int(len(feature_set) * 0.1)
        train_set, test_set = feature_set[training_size:], feature_set[:training_size]
        classifier = nltk.NaiveBayesClassifier.train(train_set)
        print('Accuracy is : ', nltk.classify.accuracy(classifier, test_set))
        return classifier
        
    # Method (private): __get_question_words_set
    # Input: None
    # Return: Set of commonly occuring words in questions
    def __get_question_words_set(self):
        question_word_list = ['what', 'where', 'when','how','why','did','do','does','have','has','am','is','are','can','could','may','would','will','should'
"didn't","doesn't","haven't","isn't","aren't","can't","couldn't","wouldn't","won't","shouldn't",'?']
        return set(question_word_list)        
    
    # Method (Public): predict_question
    # Input: Sentence to be predicted
    # Return: 1 - If sentence is question | 0 - If sentence is not question
    def predict_question(self, text):
        words = nltk.word_tokenize(text.lower())        
        if self.__get_question_words_set().intersection(words) == False:
            return 0
        if '?' in text:
            return 1
        
        features = {}
        for word in words:
            features['contains({})'.format(word.lower())] = True            
        
        prediction_result = self.classifier.classify(features)
        if prediction_result == 'whQuestion' or prediction_result == 'ynQuestion':
            return 1
        return 0
    
    # Method (Public): predict_question_type
    # Input: Sentence to be predicted
    # Return: 'WH' - If question is WH question | 'YN' - If sentence is Yes/NO question | 'unknown' - If unknown question type
    def predict_question_type(self, text):
        words = nltk.word_tokenize(text.lower())                
        features = {}
        for word in words:
            features['contains({})'.format(word.lower())] = True            
        
        prediction_result = self.classifier.classify(features)
        if prediction_result == 'whQuestion':
            return 'WH'
        elif prediction_result == 'ynQuestion':
            return 'YN'
        else:
            return 'unknown'


isQ = IsQuestion()
df_1 = pd.read_csv('queries-10k-txt', sep='\t')
df_1['is_question'] = df_1['QUERY'].apply(isQ.predict_question)
df_1['question_type'] = df_1[df_1['is_question'] == 1]['QUERY'].apply(isQ.predict_question_type)
df_1.to_csv('output/method2_output.csv', index=False)

    
        