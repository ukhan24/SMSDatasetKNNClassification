import math
import string
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# Functions
def kNNModel(k, x_train, x_test, y_train, y_test):
    # Define a model for classification
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Fit data, get prediction, and calculate accuracy
    knn.fit(x_train, y_train)
    prediction = knn.predict(x_test)
    print("Predictions: ") 
    print(prediction)
    print("")
    accuracy = accuracy_score(y_test, prediction)
    print("Prediction Accuracy: ") 
    print(accuracy)
    print("")
    
    # Get classification report ()
    clf_report = classification_report(y_test, prediction)
    print("Classification Report:")
    print(clf_report)
    
    # Create confusion matrix (using heatmap)
    createConfusionMatrix(y_test, prediction)

def createConfusionMatrix(y_actual, y_predicted):
    # Create confustion matrix object
    cm = confusion_matrix(y_actual, y_predicted)
    print("Confusion Matrix (kNN):")
    
    # Plot using heatmap
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(cm, cmap="Greens", square=True, cbar=True)   
    plt.ylabel('Actual')
    plt.xlabel('Predicted')


# ********** SMS Dataset **********
#
# Read dataset
sms_df = pd.read_csv("SMSdataset.csv")
#sms_df = sms_df[0:100]

# >>>>>>>>>> Initial Data Observations/Descriptions (data pre-analysis) <<<<<<<<<<
#
sms_general_summary = sms_df.groupby("Type").describe()
print("Data Pre-Processing Summary:")
print(sms_general_summary)
print("")
sms_types_summary = sms_df.Type.value_counts()

# Plot distribution ratio of ham to spam (gives us a better idea of the data visually)
sms_types_summary.plot.bar()
plt.title('Spam-Ham Count')

# Calculate length of each message and plot them based on spam/ham type
# (First plot is for ham and second plot is for spam messages)...
sms_df['SMS Length'] = sms_df['Message'].apply(len)
sms_df.groupby('Type').plot(kind='hist')
plt.title('SMS Length By Type')
plt.show()
 
# So far, some objective analyses of the dataset has been done using graphs and such to represent
# the data in different ways. Now let's get into how to prepare the data to be useful later on for
# classification. In other words, let's break down messages into simple words without punctuations 
# first and then convert them to lowercase messages...Finally, select the 'stopwords' to remove 
# from them

# Remove punctuations
sms_without_punctuations= []
messages_without_punctuations = []
punctuations_list = string.punctuation

for sms in sms_df['Message']:
    words_without_punctuations = [message for message in sms if message not in punctuations_list]
    sms_without_punctuations.append(words_without_punctuations)

for sms in sms_without_punctuations:
    sms_removed_punctuations = "".join(sms)
    messages_without_punctuations.append(sms_removed_punctuations)
   
# Now convert each message's words/contents to lowercase
sms_to_lowercase = []
lowercase_messages = []    

for sms in messages_without_punctuations:
    lowercase_message = [word.lower() for word in sms]
    sms_to_lowercase.append(lowercase_message)        

for sms in sms_to_lowercase:
    lowercase_message = "".join(sms)
    lowercase_messages.append(lowercase_message)

# After removing punctuations from the messages and converting all to lowercase, we now filter 
# out the recognized standard stopwords from each message, which gives us a much more useful way
# to later on classify and predict more accurately

# Remove stopwords from messages
stopwords_list = stopwords.words("english")
sms_without_stopwords_list = []
messages_without_stopwords = []

for sms in lowercase_messages:
    sms_without_stopwords = []
    words = sms.split()
   
    for word in words:
        if word not in stopwords_list:
            sms_without_stopwords.append(word)
    
    sms_without_stopwords_list.append(sms_without_stopwords)
 
for sms in sms_without_stopwords_list:
    filtered_message = " ".join(sms)
    messages_without_stopwords.append(filtered_message)
    
# Before we can go on to classification, it's better to understand our data more. In fact, 
# since the main goal is to be able to classify and predict based on the occurrence of certain
# words in our data, it's worthwhile breaking down this data to see the ratio between the most
# to least frequented words.

sms_general_summary = sms_df.groupby("Type").describe()
print("")
print("Data Post-Processing Summary:")
print(sms_general_summary)
print("")

# Let's add a column to the dataset, which is just the filtered messages (we now have a simpler
# column that displays words that we can now take into consideration, and it makes each sms easier
# to map as spam or ham) 
    
# Add new column
sms_df['Processed Messages'] = messages_without_stopwords

# Get the new length (length of each filtered message)...this is a bit useful when comparing
# to the lengths before processing/filtering the messages. In fact, just from looking at the
# first row, the length value changed from over 110 to 80, which is pretty significant

# Calculate length of processed messages
sms_df['SMS New Length'] = sms_df['Processed Messages'].apply(len)

# >>>>>>>>>> Classification Using kNN Classifier <<<<<<<<<<
#
# Define features and labels...use those to create test and train data
features = sms_df['Processed Messages']
labels = sms_df['Type']

# Convert features to useable data in classification
vectorizer = TfidfVectorizer("english")
features = vectorizer.fit_transform(features)

# Split at 80% : 20% for test and train data
sms_x_train, sms_x_test, sms_y_train, sms_y_test = train_test_split(features, labels, test_size=0.2)

# Use kNN classifier
kNNModel(100, sms_x_train, sms_x_test, sms_y_train, sms_y_test)