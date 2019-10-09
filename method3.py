import sklearn
import pandas as pd
import re
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class IsQuestionAdvanced():
    
    # Init constructor
    # Input: Type of classification: 'MNB' - Multinomial Naive Bayes | 'SVM' - Support Vector Machine
    def __init__(self, classification_type):
        self.classification_type = classification_type
        df = self.__get_data()
        df = self.__clean_data(df)
        df = self.__label_encode(df)
        vectorizer_classifier = self.__create_classifier(df, self.classification_type)
        if vectorizer_classifier is not None:
            self.vectorizer = vectorizer_classifier['vectorizer']
            self.classifier = vectorizer_classifier['classifier']        
        
    # Method (Private):  __clean_data
    # Input: Raw input dataframe
    # Processing: 1. Rename column 
    # 2. lowercase text
    # 3. preserve alpha numeric characters, whitespace, apostrophe
    # 4. filter dataframe with questiin types - what, who, when, affirmation, unknown
    # Return: Processed filtered dataframe
    def __clean_data(self, df):
        df.rename(columns={0: 'text', 1: 'type'}, inplace=True)
        df['type'] = df['type'].str.strip()
        df['text'] = df['text'].apply(lambda x: x.lower())
        df['text'] = df['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s\']','',x)))
        return df[(df['type'] == 'what') | (df['type'] == 'who') | (df['type'] == 'when') | (df['type'] == 'unknown') | (df['type'] == 'affirmation')]
    

    # Method (Private): __label_encode
    # Input: Processed dataframe
    # Processing: Use label encoding to convert text label to integer label and add it to a new column
    # Return: Processed dataframe with label encoding column
    def __label_encode(self, df):
        self.le = preprocessing.LabelEncoder()
        self.le.fit(df['type'])
        df['label'] = list(self.le.transform(df['type']))
        return df
    
    # Method (Private): __create_classifier
    # Input: 1. Processed dataframe 2. Type of classification
    # Processing: 1. Perform TFIDF Vectorization
    # 2. Appy fit_tranform using TFIDF on text column
    # 3. Split data into 70% training and 30% testing
    # 4. Perform Multinomial Naive Bayes OR SVM classifcation based on input provided
    # 5. Peform prediction for both classification techniques on test data
    # 6. Show confusion matrix and accuracy
    # Return: Dict - TFIDF Vetctorizer, Classifier    
    def __create_classifier(self, df, classification_type):
        v = TfidfVectorizer(analyzer='word',lowercase=True)
        X = v.fit_transform(df['text'])
        X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.30)
        if classification_type == 'MNB':
            clf = MultinomialNB()
            clf.fit(X_train,y_train)
            preds = clf.predict(X_test)
            print(classification_report(preds,y_test))
            print('Accuracy is: ', clf.score(X_test,y_test))
            return {'vectorizer': v, 'classifier': clf}
        elif classification_type == 'SVM':
            clf_svm = SVC(kernel='linear')
            clf_svm.fit(X_train,y_train)
            preds = clf_svm.predict(X_test)
            preds = print(classification_report(preds,y_test))
            print('Accuracy is: ', clf_svm.score(X_test,y_test))
            return {'vectorizer': v, 'classifier': clf_svm}
        else:
            print("Wrong classification type: \n Type 'MNB' - Multinomial Naive Bayes \n Type 'SVM' - Support Vector Machine")    
            

    # Method (Private): __get_data
    # Processing: Get the sample input data used to create traning, test, vectorizer, classifier data
    # Return: Pandas dataframe
    def __get_data(self):
        return pd.read_csv('/Users/kartiknagre/Downloads/sample.txt', sep=',,,', header=None)
    
    # Method (Public): predict
    # Input: An unknown new sentence
    # Return: Prediction - Typpe of question 'what', 'when', 'who'
    def predict(self, sentence):
        ex = self.vectorizer.transform([sentence])
        return list(self.le.inverse_transform(self.classifier.predict(ex)))[0]


obj = IsQuestionAdvanced('SVM')

# Run on output of first method
df_method1_out = pd.read_csv('output/method1_output.csv')
df_method1_out = df_method1_out[df_method1_out['is_question'] == 1]
df_method1_out['question_type'] = df_method1_out['QUERY'].apply(obj.predict)
df_method1_out.to_csv('output/method3_output_1.csv', index=False)

# Run on output of first method
df_method2_out = pd.read_csv('output/method2_output.csv')
del df_method2_out['question_type']
df_method2_out = df_method2_out[df_method2_out['is_question'] == 1]
df_method2_out['question_type'] = df_method2_out['QUERY'].apply(obj.predict)
df_method2_out.to_csv('output/method3_output_2.csv', index=False)



    