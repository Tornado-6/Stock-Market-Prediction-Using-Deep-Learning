
# coding: utf-8

# In[ ]:


#importing the pandas package
import pandas as pd

#reading the dataset into dataframe
data = pd.read_csv("/Users/T/Desktop/ML/Stock-Market-Prediction-Using-Deep-Learning/News/input_data.csv",header=0)
print(data.head(0))

train = data.iloc[:,2]
print(train[1])

# Using TextBlob for NLP and Sentiment Analysis


from textblob import TextBlob

processed = []

for text in train:
    blob = TextBlob(text)
    blob.tags
    blob.noun_phrases

    sum = 0.0
    count = 0
    for sentence in blob.sentences:
        sum+=sentence.sentiment.polarity
        count+=1
        print(sentence.sentiment.polarity)
    sum/=count
    print(text)
    print("Sentiment Polarity : ",str(sum))
    print("-------------------------------------------------")



    # Copying the results to a pandas dataframe with ["ID","Sentiment"]
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

    # Using pandas to write the output
    output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)

#To remove HTML Markup, we'll use Beautiful Soup
from bs4 import BeautifulSoup

#Initializing BS on a single Movie Review Object
rev1 = BeautifulSoup(train[0],"lxml")

#Printing the raw review and corrected text for cmp :
print(train[0])
print("\n-------- HTML MARKUP REMOVED ---------\n")
print(rev1.get_text())



#To tackle numbers and punctuation : re {Regular Expression} library is used
import re
letters_only = re.sub("[^a-zA-Z]"," ", rev1.get_text())
print(letters_only)


# In[ ]:


#Converting text to lowercase
lower_case = letters_only.lower()

#Tokenizing or Splitting the Words
words = lower_case.split()
print(words)


# In[ ]:


#Stop words : Words with negligible or no meaning
#To remove Stop words, we'll use NLTK package by downloading the stop words library
import nltk
nltk.download('stopwords')
#using nltk to fetch and remove the list of stop words
from nltk.corpus import stopwords
print(stopwords.words("english"))

#Remove these words from the tokenized set
words = [w for w in words if not w in stopwords.words("english")]
print(words)


# In[ ]:


#Review Cleaning function : 
def rev_to_words(raw_review):
    #Bye-Bye HTML Markups !
    review_text = BeautifulSoup(raw_review,"lxml").get_text()
    #Bye-Bye Non-Letters !
    alpha_only = re.sub("[^a-zA-Z]"," ", review_text)
    #GetLow
    words = alpha_only.lower().split()
    #Optimizing is the Goal, List to Set of stopwords for faster Search
    stops = set(stopwords.words("english"))
    #Bye-Bye Stop Words, Thanks for Stopping By !
    final_words = [w for w in words if not w in stops]
    #Fusing the words into a single space separated string
    return( " ".join(final_words))


# In[ ]:


clean_review = rev_to_words(train[0])
print(clean_review)


# In[ ]:


#Cleaning the entire training set

#Getting the Number of Reviews
num_rev = train.size


# In[ ]:


#Empty list to hold the cleaned reviews
clean_train_reviews = []

#Loop through the range of reviews
print("Cleaning and parsing the training set movie reviews")
for i in range(0, num_rev):
    if((i+1)%1000==0):
        print("Review",(i+1),"/",num_rev,"\n")
    clean_train_reviews.append(rev_to_words(train[i]))

print("Extracting features and creating bag of words\n")

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000)

# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of 
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

#Converting to arrays for easier manipulation
train_data_features = train_data_features.toarray()
print(train_data_features.shape)


# In[ ]:


# We will check the vocabulary
vocab = vectorizer.get_feature_names()
print(vocab)


# In[ ]:


#Summing up the counts of each vocab word
import numpy as np
dist = np.sum(train_data_features, axis=0)
print(dist)


# In[ ]:


#Associating each word with its count by zipping it
for tag, count in zip(vocab, dist):
    print(tag," : ",count)


# In[ ]:




#Using Random Forest for Supervised learning 
print("Random Forest being trained...")
from sklearn.ensemble import RandomForestClassifier

#Intitializing with 100 trees
forest = RandomForestClassifier(n_estimators=100)

#Fitting the forest to the training data
forest = forest.fit(train_data_features, train["sentiment"])


# In[ ]:


print("Training Successful.")
#Testing the Trained Model over TestData.tsv
test = pd.read_csv("../input/testData.tsv",sep='\t',header=0,quoting=3)

#Verifying that there are correct number of rows and columns
print(test.shape)


# In[ ]:


#Creating an empty list for appending the clean reviews
num_reviews = len(test["review"])
clean_test_reviews = []

print("Cleaning and parsing the training set movie reviews")
for i in range(0, num_rev):
    if((i+1)%1000==0):
        print("Review",(i+1),"/",num_rev,"\n")
    clean_review = rev_to_words(test["review"][i])
    clean_test_reviews.append(clean_review)

print("Extracting features and creating bag of words\n")
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

#Using the trained Random Forest for predicting sentiments : 
result = forest.predict(test_data_features)



#Copying the results to a pandas dataframe with ["ID","Sentiment"]
output = pd.DataFrame(data={"id":test["id"],"sentiment":result})

#Using pandas to write the output
output.to_csv("Bag_of_Words_model.csv", index=False, quoting=3)

