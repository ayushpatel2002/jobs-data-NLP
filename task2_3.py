#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Ayush Kamleshbhai Patel
# #### Student ID: 3891013
# 
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# 
# ## Introduction
# ### Task 2: Generating Feature Representations for Job Advertisement Descriptions
# 
# - **Objective**: Produce various feature representations for job advertisements, focusing only on the job description.
# - **Features**:
#   - **Bag-of-words model**:
#     - Generate Count vector representation for each job advertisement description based on the vocabulary from Task 1.
#   - **Models based on word embeddings**:
#     - Use an embedding language model (e.g., FastText, GoogleNews300, Word2Vec, or Glove).
#     - Build both TF-IDF weighted and unweighted vector representations for each job advertisement description using the selected language model.
# - **Output File**: 
#   - `count_vectors.txt`: Contains the sparse count vector representation of job advertisement descriptions. Each line corresponds to one advertisement in a specified format.
# 
# ### Task 3: Job Advertisement Classification
# 
# - **Objective**: Construct machine learning models to classify the category of a job advertisement.
# - **Experiments**:
#   - **Q1: Language model comparisons**:
#     - Evaluate which previously built language model performs best with the chosen machine learning model.
#   - **Q2: Does more information provide higher accuracy?**:
#     - Investigate the effect of using different features:
#       1. Only the title of the job advertisement.
#       2. Only the description (already done in Task 3).
#       3. Both the title and description combined.
# - **Evaluation**:
#   - Models' performance should be evaluated using 5-fold cross validation.

# ## Importing libraries 

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import FastText
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


# In[2]:


# logging for event tracking
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# ## Task 2. Generating Feature Representations for Job Advertisement Descriptions

# #### Loading the data

# In[3]:


# reading the labels file
df = pd.read_csv('jobs_data.csv')
df.head()


# Readung the job advertisement text, and creating another list to store the tokenized version of the advertisements('title' and 'description') text accordingly.

# In[4]:


txt_fname = 'description.txt'
with open(txt_fname) as txtf:
    description = txtf.read().splitlines() # reading a list of strings, each for a document/article
    titles = txtf.read().splitlines()
tk_description = [a.split(' ') for a in description]
tk_title = [t.split(' ') for t in description]


# In[5]:


titles_fname = 'title.txt'
with open(txt_fname) as txtf:
    titles = txtf.read().splitlines()
tk_title = [t.split(' ') for t in description]
import string
tk_title = [list(map(lambda term: term.lower().translate(str.maketrans('', '', string.punctuation)), doc)) for doc in tk_title]


# In[6]:


df['Description'] = description
df['Tokenized Description'] = tk_description
df['Tokenized Title'] = tk_title
df.sample(n = 5) # look at a few examples


# In[7]:


words = list(chain.from_iterable(tk_description))
# set of unique words
vocab = sorted(list(set(words)))
# total number


# ### Bag Of Words Model

# <h3 style = "color: #00FF00"> Creating Count Vectors

# <h4 style="color: #4169E1">The following code cell generates Count vector 
# representation for each job advertisement description using the FastText model

# In[8]:


joined_description = [' '.join(review) for review in tk_description]
cVectorizer = CountVectorizer(analyzer = "word",vocabulary = vocab) # initialised the CountVectorizer
count_features = cVectorizer.fit_transform(joined_description)
count_features.shape


# In[9]:


count_features_df = pd.DataFrame(count_features.toarray(), columns=cVectorizer.get_feature_names_out())
# print out samples
count_features_df.head(3)


# <h3 style = "color: #00FF00"> Saving it into the file

# In[10]:


webindex = df['Webindex']
count_features = cVectorizer.fit_transform(joined_description).toarray()

def save_count_vector(count_features, webindex, filename):
    with open(filename, 'w') as f:
        for i in range(len(count_features)):
            f.write('#' + str(webindex[i]) + ',')
            for j in range(len(count_features[i])):
                if count_features[i][j] != 0:
                    f.write(str(j) + ':' + str(count_features[i][j]) + ',')
            f.write('\n')
    f.close()
    print('Count vector representation saved to ' + filename)


save_count_vector(count_features, webindex, 'count_vectors.txt')


# <h3 style = "color: #00FF00"> Creating TF-IDF Vectors

# In[11]:


joined_description = [' '.join(review) for review in tk_description]
tVectorizer = TfidfVectorizer(analyzer = "word",vocabulary = vocab) # initialised the TfidfVectorizer
tfidf_features = tVectorizer.fit_transform(joined_description) # generate the tfidf vector representation for all job description
tfidf_features.shape


# In[12]:


def write_vectorFile(data_features,filename):
    num = data_features.shape[0] # the number of document
    out_file = open(filename, 'w') # creates a txt file and open to save the vector representation
    for a_ind in range(0, num): # loop through each job description by index
        for f_ind in data_features[a_ind].nonzero()[1]: # for each word index that has non-zero entry in the data_feature
            value = data_features[a_ind][0,f_ind] # retrieve the value of the entry from data_features
            out_file.write("{}:{} ".format(f_ind,value)) # write the entry to the file in the format of word_index:value
        out_file.write('\n') # start a new line after each job description
    out_file.close() # close the file  


# In[13]:


tfidf_features_file = "jobs_data_vector.txt" # file name of the tfidf vector

write_vectorFile(tfidf_features,tfidf_features_file) # write the tfidf vector to file


# <h2 style="color: #FFFF00"> Creating Model based on word embeddings

# <h3 style = "color: #00FF00"> Creating and Saving <b>FastText</b> Model

# The functions below will help us to carry out the remaining tasks of the requirements.
# 
# ### Summary of Functions:
# 
# 1. **docvecs**: 
#    - Generates document embeddings by averaging the word embeddings of the words present in the document.
#    
# 2. **plotTSNE**:
#    - Provides a t-SNE visualization of document embeddings. This is useful for visualizing the distribution and clustering of documents in a 2D space.
# 
# 3. **weighted_docvecs**:
#    - Generates document embeddings weighted by their TF-IDF values. This gives more importance to terms that are more significant in the document.
# 
# 4. **gen_vocIndex**:
#    - Creates a dictionary that maps from vocabulary indices to their corresponding words. This helps in referencing words using their indices.
# 
# 5. **doc_wordweights**:
#    - Constructs a list of dictionaries where each dictionary represents a document. The keys in the dictionary are words, and the values are their corresponding TF-IDF weights.
# 

# In[14]:


def docvecs(embeddings, docs):
    """
    Generate document embeddings by averaging word embeddings.
    
    Args:
    - embeddings: Pre-trained word embeddings model.
    - docs: List of tokenized documents.

    Returns:
    - Numpy array of document embeddings.
    """
    vecs = np.zeros((len(docs), embeddings.vector_size))
    for i, doc in enumerate(docs):
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        docvec = np.vstack([embeddings[term] for term in valid_keys])
        docvec = np.sum(docvec, axis=0)  # Summing up word vectors for the entire document
        vecs[i,:] = docvec
    return vecs

def plotTSNE(labels,features):
    """
    Plot t-SNE visualization of document embeddings.
    
    Args:
    - labels: Pandas series of document labels.
    - features: Numpy array of document embeddings.

    Returns:
    - None (Displays a scatter plot).
    """
    categories = sorted(labels.unique())
    SAMPLE_SIZE = int(len(features) * 0.3)
    np.random.seed(3891013)
    indices = np.random.choice(range(len(features)), size=SAMPLE_SIZE, replace=False)
    projected_features = TSNE(n_components=2, random_state=3891013).fit_transform(features[indices].astype(int))
    colors = ['pink', 'green', 'midnightblue', 'orange', 'darkgrey']
    for i in range(0,len(categories)):
        points = projected_features[(labels[indices] == categories[i])]
        plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[i], label=categories[i])
    plt.title("Feature vector for each article, projected on 2 dimensions.")
    plt.legend()
    plt.show()

def weighted_docvecs(embeddings, tfidf, docs):
    """
    Generate TF-IDF weighted document embeddings.
    
    Args:
    - embeddings: Pre-trained word embeddings model.
    - tfidf: List of dictionaries with tf-idf weights for each document.
    - docs: List of tokenized documents.

    Returns:
    - Numpy array of weighted document embeddings.
    """
    vecs = np.zeros((len(docs), embeddings.vector_size))
    for i, doc in enumerate(docs):
        valid_keys = [term for term in doc if term in embeddings.key_to_index]
        tf_weights = [float(tfidf[i].get(term, 0.)) for term in valid_keys]
        weighted = [embeddings[term] * w for term, w in zip(valid_keys, tf_weights)]
        docvec = np.vstack(weighted)
        docvec = np.sum(docvec, axis=0)  # Summing up weighted word vectors for the entire document
        vecs[i,:] = docvec
    return vecs

def gen_vocIndex(voc_fname):
    """
    Generate a dictionary mapping from vocabulary indices to words.
    
    Args:
    - voc_fname: Path to the vocabulary file.

    Returns:
    - Dictionary with vocabulary index as key and corresponding word as value.
    """
    with open(voc_fname) as vocf: 
        voc_Ind = [l.split(':') for l in vocf.read().splitlines()]
    return {int(vi[1]):vi[0] for vi in voc_Ind}

# Generates the w_index:word dictionary
voc_fname = 'vocab.txt'
voc_dict = gen_vocIndex(voc_fname)

def doc_wordweights(fName_tVectors, voc_dict):
    """
    Generate a list of dictionaries with tf-idf weights for each document.
    
    Args:
    - fName_tVectors: Path to the file containing tf-idf vectors.
    - voc_dict: Dictionary with vocabulary index as key and word as value.

    Returns:
    - List of dictionaries with words as keys and their tf-idf weights as values for each document.
    """
    tfidf_weights = []
    with open(fName_tVectors) as tVecf: 
        tVectors = tVecf.read().splitlines()
    for tv in tVectors:
        tv = tv.strip()
        weights = tv.split(' ')
        weights = [w.split(':') for w in weights]
        wordweight_dict = {voc_dict[int(w[0])]:w[1] for w in weights}
        tfidf_weights.append(wordweight_dict) 
    return tfidf_weights

fName_tVectors = 'jobs_data_vector.txt'
tfidf_weights = doc_wordweights(fName_tVectors, voc_dict)


# We create a dataframe to store accuracy of different models we will use in the subsequent steps  

# In[15]:


df_results = pd.DataFrame(columns=['Model', 'Accuracy'])
performence  = pd.DataFrame(columns=['Title', 'accuracy'])


# In[16]:


# 1. Set the corpus file names/path
corpus_file = 'description.txt'

# 2. Initialise the Fast Text model
descFT = FastText(vector_size=50) 

# 3. build the vocabulary
descFT.build_vocab(corpus_file=corpus_file)

# 4. train the model
descFT.train(
    corpus_file=corpus_file, epochs=descFT.epochs,
    total_examples=descFT.corpus_count, total_words=descFT.corpus_total_words,
)

print(descFT)


# In[17]:


descFT_wv = descFT.wv
print(descFT_wv)


# In the cell below, we save the trained model. For new I have commented it out because we use the already saved and trained model 

# In[18]:


# Save the model
# descFT.save("models/FastText/descFT.model")


# We can retrieve the KeyedVectors from the model as follows

# In[19]:


descFT = FastText.load("models/FastText/descFT.model")
print(descFT)
descFT_wv= descFT.wv


# <h4 style="color: #4169E1"> The following code cell generates unweighted vector 
# representation for each job advertisement description using the FastText model

# In[20]:


descFT_dvs = docvecs(descFT_wv, df['Tokenized Description'])


# In[21]:


# explore feature space
features = descFT_dvs
plotTSNE(df['Category'],features)


# <p style="color: #FFA500">The above graph is for unweighted feature vector projection for each tokenized job description

# The following code cell: -
# 
# - Defines a `LogisticRegression` model.
# - Splits data into training and test sets.
# - Performs 5-fold cross-validation on training data:
#   - Train on each fold.
#   - Validate and compute accuracy.
# - Trains the model on the unweighted feature represntation for tokenized job descriptions
# - Evaluates on test set and display a confusion matrix.

# In[22]:


import seaborn as sns


# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=200, random_state=3891013)

# Splitting data
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(descFT_dvs, df['Category'], list(range(0,len(df))), test_size=0.33, random_state=3891013)

# Setting up the KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=3891013)
fold_num = 1
scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model.fit(X_train_fold, y_train_fold)
    y_pred_fold = model.predict(X_val_fold)
    score = model.score(X_val_fold, y_val_fold)
    
    scores.append(score)
    print(f"Fold {fold_num}: Accuracy: {score:.4f}")
    
    cm = confusion_matrix(y_val_fold, y_pred_fold)
    
    '''
    the following piece of code which is commented will generate confusion matrix for each fold
    '''
    # # Plotting
    # plt.figure(figsize=(10,7))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title(f'Confusion Matrix for Fold {fold_num}')
    # plt.show()
    
    fold_num += 1

avg_score = sum(scores) / len(scores)
print(f"Average Cross-Validation Accuracy: {avg_score:.4f}")
df_results.loc[len(df_results)] = ['unweighted vector(using in-house trained FastText)', avg_score]
performence.loc[len(performence)] = ['Unweighted Vectors for Description', avg_score]

# After the 5-fold CV, fit the model on the entire training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Entire Test Set')
plt.show()


# <h4>Average Cross-Validation Accuracy for Unweighted vector 
# representation for each job advertisement description: 0.7821

# <h3 style = "color: #00FF00"> Generating TF-IDF weighted document vectors

# In[23]:


def gen_vocIndex(voc_fname):
    with open(voc_fname) as vocf: 
        voc_Ind = [l.split(':') for l in vocf.read().splitlines()] # each line is 'index,word'
    return {int(vi[1]):vi[0] for vi in voc_Ind}


# Generates the w_index:word dictionary
voc_fname = 'vocab.txt' # path for the vocabulary
voc_dict = gen_vocIndex(voc_fname)
voc_dict


# - the `doc_wordweights` function takes the tfidf document vector file, as well as the w_index:word dictionary, creates the mapping between w_index and the actual word, and creates a dictionary of word:weight or each unique word appear in the document.

# In[24]:


def doc_wordweights(fName_tVectors, voc_dict):
    tfidf_weights = [] # a list to store the  word:weight dictionaries of documents
    
    with open(fName_tVectors) as tVecf: 
        tVectors = tVecf.read().splitlines() # each line is a tfidf vector representation of a document in string format 'word_index:weight word_index:weight .......'
    for tv in tVectors: # for each tfidf document vector
        tv = tv.strip()
        weights = tv.split(' ') # list of 'word_index:weight' entries
        weights = [w.split(':') for w in weights] # change the format of weight to a list of '[word_index,weight]' entries
        wordweight_dict = {voc_dict[int(w[0])]:w[1] for w in weights} # construct the weight dictionary, where each entry is 'word:weight'
        tfidf_weights.append(wordweight_dict) 
    return tfidf_weights

fName_tVectors = 'jobs_data_vector.txt'
tfidf_weights = doc_wordweights(fName_tVectors, voc_dict)


# In[25]:


# take a look at the tfidf word weights dictionary of the first document
tfidf_weights[0]


# <h4 style="color: #4169E1">The following code cell generates TF-IDF weighted vector 
# representation for each job advertisement description using the FastText model

# In[26]:


weighted_descFT_dvs = weighted_docvecs(descFT_wv, tfidf_weights, df['Tokenized Description'])


# And we can do very much the same thing as what we do before for other models. 
# Here, we will do this as loops, for each model:
# - we plot out the feature vectors  projected in a 2-dimensional space,then 
# - we build the logistic regression model for document classfication and report the model performance.

# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
seed = 3891013


dv = weighted_descFT_dvs
name = "Weighted In-house FastText"
print(name + ": tSNE 2 dimensional projected Feature space")
plotTSNE(df['Category'],dv)


# <p style="color: #FFA500">The above graph is for weighted feature vector projection for each tokenized job description

# The following code cell: -
# 
# - Defines a `LogisticRegression` model.
# - Splits data into training and test sets.
# - Performs 5-fold cross-validation on training data:
#   - Train on each fold.
#   - Validate and compute accuracy.
# - Trains the model on the TF-IDF weighted feature represntation for tokenized job descriptions
# - Evaluates on test set and display a confusion matrix.

# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')
seed = 3891013
import seaborn as sns
import matplotlib.pyplot as plt


# creating training and test split
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(dv, df['Category'], list(range(0, len(df))), test_size=0.33, random_state=3891013)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=200, random_state=3891013)

kf = KFold(n_splits=5, shuffle=True, random_state=3891013)
fold_num = 1
scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model.fit(X_train_fold, y_train_fold)
    y_pred_fold = model.predict(X_val_fold)
    score = model.score(X_val_fold, y_val_fold)
    
    scores.append(score)
    print(f"Fold {fold_num}: Accuracy: {score:.4f}")
    
    '''
    the following piece of code which is commented will generate confusion matrix for each fold
    '''
    # Confusion matrix for the current fold
    # cm = confusion_matrix(y_val_fold, y_pred_fold)
    # plt.figure(figsize=(10,7))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title(f'Confusion Matrix for Fold {fold_num}')
    # plt.show()
    
    fold_num += 1

avg_score = sum(scores) / len(scores)
print(f"Average Cross-Validation Accuracy: {avg_score:.4f}")
df_results.loc[len(df_results)] = ['weighted TF-IDF vector(using in-house trained FastText)', avg_score]
performence.loc[len(performence)] = ['Weighted Vectors for Description', avg_score]

# Fit the model on the entire training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Generate the confusion matrix for the entire test set
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Entire Test Set')
plt.show()


# <h4>Average Cross-Validation Accuracy for TF-IDF Weighted vector 
# representation for each job advertisement description: 0.7667

# ### Language model comparisons

# #### Based Upon Count Features

# The following code cell: -
# 
# - Defines a `LogisticRegression` model.
# - Splits data into training and test sets.
# - Performs 5-fold cross-validation on training data:
#   - Train on each fold.
#   - Validate and compute accuracy.
# - Trains the model on the Count vector 
# representation for tokenized job descriptions
# - Evaluates on test set and display a confusion matrix.

# In[29]:


model = LogisticRegression(max_iter=200, random_state=3891013)

# Splitting data
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(count_features, df['Category'], list(range(0,len(df))), test_size=0.33, random_state=3891013)

# Setting up the KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=3891013)
fold_num = 1
scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model.fit(X_train_fold, y_train_fold)
    y_pred_fold = model.predict(X_val_fold)
    score = model.score(X_val_fold, y_val_fold)
    
    scores.append(score)
    print(f"Fold {fold_num}: Accuracy: {score:.4f}")
    
    cm = confusion_matrix(y_val_fold, y_pred_fold)
    
    '''
    the following piece of code which is commented will generate confusion matrix for each fold
    '''
    # # Plotting
    # plt.figure(figsize=(10,7))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title(f'Confusion Matrix for Fold {fold_num}')
    # plt.show()
    
    fold_num += 1

avg_score = sum(scores) / len(scores)
print(f"Average Cross-Validation Accuracy: {avg_score:.4f}")
df_results.loc[len(df_results)] = ['Count Features', avg_score]

# After the 5-fold CV, fit the model on the entire training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Entire Test Set for Count Features')
plt.show()


# <h4>Average Cross-Validation Accuracy Count vector 
# representation each job advertisement description: 0.8515

# <h3 style="color: #FFA500">Answer for Q1: Language model comparisons

# In[30]:


df_results


# In[31]:


ax = df_results.plot(x='Model', y='Accuracy', kind='bar', legend=False, figsize=(10,6), width=0.4)

# Adjust the y-axis limits to focus on a range that highlights differences
# Considering the data provided, setting the range from 0.75 to 0.90
ax.set_ylim([0.75, 0.9])

# Add title and labels
plt.title('Accuracy for Different Titles')
plt.ylabel('Accuracy')
plt.xlabel('Title')
plt.xticks(rotation=45, ha='right')

# Display y-values with 4 decimal places on the y-axis
ax.yaxis.set_major_formatter('{:.4f}'.format)

# Show the plot
plt.tight_layout()
plt.show()


# <h4 style="color: #00BFFF"> From the above graph, we can clearly see that we get highest accuracy for <b>Count vector representation</b> which is around 0.85 or 85 percent.
# 
# The possible reason for the oberserved output could be: -
# - **Frequency Matters**: In job descriptions, important terms are often repeated for emphasis or clarity. Count Vector Representation captures this repetition.
# - **Specificity of Keywords**: Job descriptions tend to have field-specific jargon or keywords that are strong indicators of the job field. Count vectors can effectively capture these unique terms.
# - **High Dimensionality**: Count vectors are high-dimensional, capturing a wide array of terms from the corpus. This granularity can be beneficial for distinguishing between nuanced job fields.
# 

# <h3 style="color: #FFA500">Answer for Q2: Does more information provide higher accuracy?

# <h3 style = "color: #00FF00"> Creating Models Based on Title

# ##### Testing Model on unweighted Title

# The following code cell generates unweighted vector 
# representation for each job advertisement Title using the FastText model

# In[32]:


descFT_dvs_titles = docvecs(descFT_wv, df['Tokenized Title'])


# The following code cell: -
# 
# - Defines a `LogisticRegression` model.
# - Splits data into training and test sets.
# - Performs 5-fold cross-validation on training data:
#   - Train on each fold.
#   - Validate and compute accuracy.
# - Trains the model on the unweighted tokenized title vectors for each job advertisement
# - Evaluates on test set and display a confusion matrix.

# In[33]:


# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=200, random_state=3891013)

# Splitting data
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(descFT_dvs_titles, df['Category'], list(range(0,len(df))), test_size=0.33, random_state=3891013)

# Setting up the KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=3891013)
fold_num = 1
scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model.fit(X_train_fold, y_train_fold)
    y_pred_fold = model.predict(X_val_fold)
    score = model.score(X_val_fold, y_val_fold)
    
    scores.append(score)
    print(f"Fold {fold_num}: Accuracy: {score:.4f}")
    
    cm = confusion_matrix(y_val_fold, y_pred_fold)
    
    '''
    the following piece of code which is commented will generate confusion matrix for each fold
    '''
    # # Plotting
    # plt.figure(figsize=(10,7))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title(f'Confusion Matrix for Fold {fold_num}')
    # plt.show()
    
    fold_num += 1

avg_score = sum(scores) / len(scores)
print(f"Average Cross-Validation Accuracy: {avg_score:.4f}")
performence.loc[len(performence)] = ['Unweighted Vectors for Title', avg_score]

# After the 5-fold CV, fit the model on the entire training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Entire Test Set for Titles')
plt.show()


# <h4>Average Cross-Validation Accuracy for unweighted vector 
# representation for each job advertisement Title: 0.7899

# ##### Testing Model on weighted Title

# The following code cell generates TF-IDF weighted vector 
# representation for each job advertisement Title using the FastText model

# In[34]:


weighted_descFT_dvs_title = weighted_docvecs(descFT_wv, tfidf_weights, df['Tokenized Title'])


# The following code cell: -
# 
# - Defines a `LogisticRegression` model.
# - Splits data into training and test sets.
# - Performs 5-fold cross-validation on training data:
#   - Train on each fold.
#   - Validate and compute accuracy.
# - Trains the model on the TF-IDF weighted tokenized title vectors for each job advertisement
# - Evaluates on test set and display a confusion matrix.

# In[35]:


# creating training and test split
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(weighted_descFT_dvs_title, df['Category'], list(range(0, len(df))), test_size=0.33, random_state=3891013)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=200, random_state=3891013)

kf = KFold(n_splits=5, shuffle=True, random_state=3891013)
fold_num = 1
scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model.fit(X_train_fold, y_train_fold)
    y_pred_fold = model.predict(X_val_fold)
    score = model.score(X_val_fold, y_val_fold)
    
    scores.append(score)
    print(f"Fold {fold_num}: Accuracy: {score:.4f}")
    
    '''
    the following piece of code which is commented will generate confusion matrix for each fold
    '''
    # Confusion matrix for the current fold
    # cm = confusion_matrix(y_val_fold, y_pred_fold)
    # plt.figure(figsize=(10,7))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title(f'Confusion Matrix for Fold {fold_num}')
    # plt.show()
    
    fold_num += 1

avg_score = sum(scores) / len(scores)
print(f"Average Cross-Validation Accuracy: {avg_score:.4f}")
performence.loc[len(performence)] = ['Weighted Vectors for Title', avg_score]

# Fit the model on the entire training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Generate the confusion matrix for the entire test set
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Entire Test Set')
plt.show()


# <h4>Average Cross-Validation Accuracy for TF-IDF weighted vector 
# representation for each job advertisement Title: 0.7629

# <h3 style = "color: #00FF00"> Creating Model based on Title+Description

# ##### Testing Model on **unweighted** Title+Description

# The following code cell generates unweighted vector 
# representation for each job advertisement Title+Description using the FastText model

# In[36]:


descFT_dvs_titles_and_desc = docvecs(descFT_wv, df['Tokenized Title']+df['Tokenized Description'])


# The following code cell: -
# 
# - Defines a `LogisticRegression` model.
# - Splits data into training and test sets.
# - Performs 5-fold cross-validation on training data:
#   - Train on each fold.
#   - Validate and compute accuracy.
# - Trains the model on the unweighted tokenized title+description(concatinated) vectors for each job advertisement
# - Evaluates on test set and display a confusion matrix.

# In[37]:


# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=200, random_state=3891013)

# Splitting data
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(descFT_dvs_titles_and_desc, df['Category'], list(range(0,len(df))), test_size=0.33, random_state=3891013)

# Setting up the KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=3891013)
fold_num = 1
scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model.fit(X_train_fold, y_train_fold)
    y_pred_fold = model.predict(X_val_fold)
    score = model.score(X_val_fold, y_val_fold)
    
    scores.append(score)
    print(f"Fold {fold_num}: Accuracy: {score:.4f}")
    
    cm = confusion_matrix(y_val_fold, y_pred_fold)
    
    '''
    the following piece of code which is commented will generate confusion matrix for each fold
    '''
    # # Plotting
    # plt.figure(figsize=(10,7))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title(f'Confusion Matrix for Fold {fold_num}')
    # plt.show()
    
    fold_num += 1

avg_score = sum(scores) / len(scores)
print(f"Average Cross-Validation Accuracy: {avg_score:.4f}")
performence.loc[len(performence)] = ['Unweighted Vectors for Title and Description Combined', avg_score]

# After the 5-fold CV, fit the model on the entire training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Entire Test Set for Titles')
plt.show()


# <h4>Average Cross-Validation Accuracy for unweighted vector 
# representation for each job advertisement Title+Description: 0.7783

# ##### Testing Model on **weighted** Title+Description

# The following code cell generates TF-IDF weighted vector 
# representation for each job advertisement Title+Description using the FastText model

# In[38]:


weighted_descFT_dvs_titles_and_desc = weighted_docvecs(descFT_wv, tfidf_weights, df['Tokenized Title']+df['Tokenized Description'])


# The following code cell: -
# 
# - Defines a `LogisticRegression` model.
# - Splits data into training and test sets.
# - Performs 5-fold cross-validation on training data:
#   - Train on each fold.
#   - Validate and compute accuracy.
# - Trains the model on the TF-IDF weighted tokenized title+description(concatinated) vectors for each job advertisement
# - Evaluates on test set and display a confusion matrix.

# In[39]:


# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=200, random_state=3891013)

# Splitting data
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(weighted_descFT_dvs_titles_and_desc, df['Category'], list(range(0,len(df))), test_size=0.33, random_state=3891013)

# Setting up the KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=3891013)
fold_num = 1
scores = []

for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
    model.fit(X_train_fold, y_train_fold)
    y_pred_fold = model.predict(X_val_fold)
    score = model.score(X_val_fold, y_val_fold)
    
    scores.append(score)
    print(f"Fold {fold_num}: Accuracy: {score:.4f}")
    
    cm = confusion_matrix(y_val_fold, y_pred_fold)
    
    '''
    the following piece of code which is commented will generate confusion matrix for each fold
    '''
    # # Plotting
    # plt.figure(figsize=(10,7))
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    # plt.xlabel('Predicted labels')
    # plt.ylabel('True labels')
    # plt.title(f'Confusion Matrix for Fold {fold_num}')
    # plt.show()
    
    fold_num += 1

avg_score = sum(scores) / len(scores)
print(f"Average Cross-Validation Accuracy: {avg_score:.4f}")
performence.loc[len(performence)] = ['weighted Vectors for Title and Description Combined', avg_score]

# After the 5-fold CV, fit the model on the entire training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plotting
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix for Entire Test Set for Titles')


# <h4>Average Cross-Validation Accuracy for TF-IDF weighted vector 
# representation for each job advertisement Title+Description: 0.7706

# In[40]:


ax = performence.plot(x='Title', y='accuracy', kind='bar', legend=False, figsize=(10,6), width=0.4)

# Adjust the y-axis limits to focus on a range that highlights differences
# Considering the data provided, setting the range from 0.75 to 0.8
ax.set_ylim([0.75, 0.8])

# Add title and labels
plt.title('Accuracy for Different Titles')
plt.ylabel('Accuracy')
plt.xlabel('Title')
plt.xticks(rotation=45, ha='right')

# Display y-values with 4 decimal places on the y-axis
ax.yaxis.set_major_formatter('{:.4f}'.format)

# Show the plot
plt.tight_layout()
plt.show()


performence


# <h4 style="color: #00BFFF">
# Analysis of Job Advertisement Title and Description on Model Accuracy
# 
# **Results**:
# - **Unweighted Vectors**:
#   - Description Alone: `78.21%`
#   - Title Alone: `78.99%`
#   - Title + Description: `77.83%`
# 
# - **Weighted Vectors**:
#   - Description Alone: `76.67%`
#   - Title Alone: `76.29%`
#   - Title + Description: `77.06%`
# 
# **Discussion**:
# 1. The title on its own slightly outperforms the description when using unweighted vectors.
# 2. With weighted vectors, the description has a slight edge over the title.
# 3. Combining title and description does not lead to an accuracy boost. In fact, the accuracy drops slightly compared to using just the title or description alone.
# 
# **Conclusion**:
# Using the title of a job advertisement does not result in a significant accuracy boost when combined with the description, based on the current method of combination. However, the title alone possesses a good predictive capability that might be explored further.
# 
# 
# **Probable Reasons for Observed Results**
# 
# 1. **Redundancy**: Combining titles and descriptions might repeat information.
# 2. **Dilution**: Important features in titles could get diluted by longer descriptions.
# 3. **Different Structures**: Titles and descriptions might have distinct linguistic patterns, introducing noise when combined.
# 4. **Representation Limitation**: The method of vector representation might not capture the combined semantics effectively.
# 5. **Overfitting**: Combining could lead to overfitting, especially on smaller datasets.
# 6. **Merging Strategy**: Direct concatenation might not be optimal for combining the information.
# 
# In essence, while both pieces of data are valuable, the method of merging is critical.
# 
# 

# ## Summary
# - **Bag Of Words Model**: The notebook moves on to the Bag Of Words model. The code assigns descriptions and tokenized versions of the descriptions and titles to the dataframe.
# - **Creating Binary Vectors**: Binary vectors are created by extracting a set of unique words from the descriptions and forming a vocabulary from them.
# - **Creating Count Vectors**: Count vectors are created for the job advertisements. The tokenized descriptions are joined, and the CountVectorizer is initialized with binary set to True, using the generated vocabulary.
# - **Generating Count Vector Representation using the FastText Model**: Count vector representations are generated for each job advertisement description using the FastText model. The CountVectorizer is initialized using the vocabulary, and the feature shape is displayed.
# - **Saving Count Vectors to File**: The count features obtained from the CountVectorizer are transformed into a dataframe and displayed.
# - **Creating TF-IDF Vectors**: The TF-IDF vectors for the job advertisements are created. Webindex is extracted from the dataframe, and the TfidfVectorizer is initialized with the vocabulary. The shape of the tfidf features is also displayed.
# - **Creating Model based on Word Embeddings**: The focus shifts to creating models based on word embeddings. The TfidfVectorizer is initialized with the vocabulary, and tfidf vector representations are generated for all job descriptions.
# - **Generating Unweighted Vector Representation with FastText**: The FastText model's word vectors are accessed and stored in `descFT_wv`.
# - **Graph for Unweighted Feature Vector Projection**: A comment indicates that there's a graph visualizing unweighted feature vector projections for tokenized job descriptions. The code hints at saving the model, but the actual saving action is commented out.
# - **Defining and Evaluating Logistic Regression Model (TF-IDF Weighted Vectors)**: A logistic regression model is defined, with data split into training and test sets. A 5-fold cross-validation is employed on the training data, wherein each fold is trained, validated, and the accuracy determined. The model is trained on TF-IDF weighted feature representations of the tokenized job descriptions, and its performance is evaluated on the test set, including a confusion matrix.
# - **Generating TF-IDF Weighted Document Vectors**: Document vectors are generated using the FastText model's word vectors and the tokenized descriptions. This results in TF-IDF weighted vectors for each job advertisement description.
# - **Utility Function for Word Weights**: The `doc_wordweights` function is mentioned, which creates a mapping between word indices and actual words, subsequently creating a dictionary of word:weight for each unique word appearing in the document.
# - **Generating TF-IDF Weighted Vector with FastText**: The cell mentions the generation of TF-IDF weighted vector representations for job advertisement descriptions using the FastText model.
# - **Looping Through Models**: The notebook suggests looping through different models to visualize feature vectors projected in a 2-dimensional space, evaluate the model's performance, and display the accuracy.
# - **Answer for Q1: Language Model Comparisons**: The notebook provides an answer or discussion related to a question on language model comparisons. A logistic regression model is defined.   The following was performed for answering Q1: -  
#     - **Analysis of Model Accuracy**: An analysis is conducted regarding the accuracy of the model, especially with the count vector representation. The analyses suggests that the highest accuracy was achieved using count vector representation, approximated at 85%.
# - **Answer for Q2: Does More Information Provide Higher Accuracy?**: The notebook provides an answer or discussion for a question about the correlation between the amount of information and model accuracy. A bar plot is generated to visualize the accuracy of different models.    The following was performed for answering Q2:  
#     - **Creating Models Based on Title**: The notebook shifts its focus to creating models based on job titles. The `docvecs` function is used to generate document vectors for the tokenized titles using the FastText model's word vectors.
#     - **Testing Model on Unweighted Title**: A Logistic Regression model is initialized, and the data is split into training and test sets. The model is then trained on unweighted tokenized title vectors for each job advertisement.
#     - **Generating Unweighted Vector Representation for Job Advertisement Titles**: Unweighted vector representations are generated for each job advertisement title using the FastText model.
#     - **Defining and Evaluating Logistic Regression Model (Unweighted Title Vectors)**: The same approach as before is applied, but this time the model is trained on unweighted tokenized title vectors for each job advertisement.
#     - **Testing Model on Weighted Title**: A brief mention of testing a model on weighted titles.
#     - **Generating TF-IDF Weighted Vector Representation for Job Advertisement Titles**: TF-IDF weighted vector representations are generated for each job advertisement title using the FastText model.
#     - **Combining Titles and Descriptions for Model Evaluation**: The notebook hints at combining tokenized titles and descriptions to generate document vectors and evaluate a model based on this combined information.
#     - **Further Analysis and Conclusions**: While not explicitly mentioned in the bullet points provided, the remaining cells of the notebook likely involve further analyses, discussions, and conclusions based on the tasks and models evaluated.
# 
# 
