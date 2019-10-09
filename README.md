# nlp-question-detection

### Given a sentence, predict if the sentence is a question or not

I have used 3 methods

#### Part 1 - detect a question and 
#### Part 2 - detect the type of the question.

## METHOD 1: Using Basic Parse Tree created using Stanford's CORE NLP
This is the most basic experiment among the three.

I have used the Penn Treebank’s Clause level tags to detect if the sentence is a question.
I specifically check for occurence of two tags:

#### `SBARQ` - Direct question introduced by a wh-word or a wh-phrase. Indirect questions and
relative clauses should be bracketed as SBAR, not SBARQ.

#### `SQ` - Inverted yes/no questions, or main clause of a wh-question, following the wh-phrase in SBARQ.
If the parse tree of the sentence contains either of the two tags then it is classified as a
question.
The legend is:
0 - Not a question
1 - Is a question
The results are:
0 ( Not a question) - 9332
1 (Is a question) - 668

## METHOD 2: Classification using NLTK’s Multinomial Naive Bayes

In this method I use nps_chat from nltk.corpus as the training data.
There are 10567 posts in the corpus which includes label. I was particularly interested in two labels ‘whQuestion’ and ‘ynQuestion’.

I used a boolean vectorizer to train the data and tested it on test data and train test split on a model generated using Multinomial Naive Bayes classifier.

The model gave an accuracy of `67%`
Upon running this model against the given unseen data, the following results are obtained:
The legend is:
0 - Not a question
1 - Is a question
The results are:
0 ( Not a question) - 8799
1 (Is a question) - 1201

For Part 2 , which is to identify the question subtypes, I use the same model and run it on the 1201 sentences which are classified as questions.

Now instead of classifying the sentence as question or not question I classified the question as WH
questions and Yes/No questions.
Remember, that these were the two labels part of the training data retrieved from nps_chat

The legend is:
WH - WH question
YN - Yes/No question
The results are:
WH - 944
YN - 257

# METHOD 3: Advanced Classification using Sklean’s Multinomial Naive
Bayes and Support Vector Machine
I used this technique mainly for Part 2 - to determine the subtypes of questions.

To improve on the performance from method 2, I decided to perform some advanced classification.
I retrieved training data from an external source which includes 1483 sentences which is labeled as what, who, when, affirmation, unknown.

This training data is available in sample.txt

What - what questions
Who - who questions
When - when questions
Affirmation - yes/no questions
Unknown - Unknown type questions.

Later I used TF-IDF as vectorization technique to prepare the training data.

After training test split (70/30) I achieved an accuracy of `73%` with Multinomial Naive Bayes and
`97%` with SVM using linear kernel.

The SVM model performed particularly well with corner case question such as 
`What time is the train leaving tomorrow ?` -> When question rather than What as it pertains to time.
