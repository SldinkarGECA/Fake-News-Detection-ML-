# Detection Fake News Recognition System

What is Fake News: Fake news generally spread through social media and other online media

# Importing libraries


```python
import numpy as np
import pandas as pd
import itertools
```

# Reading the data


```python
df = pd.read_csv('news.csv')
# based on political dataset
```


```python
df.shape
```




    (6335, 4)




```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8476</td>
      <td>You Can Smell Hillary’s Fear</td>
      <td>Daniel Greenfield, a Shillman Journalism Fello...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10294</td>
      <td>Watch The Exact Moment Paul Ryan Committed Pol...</td>
      <td>Google Pinterest Digg Linkedin Reddit Stumbleu...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3608</td>
      <td>Kerry to go to Paris in gesture of sympathy</td>
      <td>U.S. Secretary of State John F. Kerry said Mon...</td>
      <td>REAL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10142</td>
      <td>Bernie supporters on Twitter erupt in anger ag...</td>
      <td>— Kaydee King (@KaydeeKing) November 9, 2016 T...</td>
      <td>FAKE</td>
    </tr>
    <tr>
      <th>4</th>
      <td>875</td>
      <td>The Battle of New York: Why This Primary Matters</td>
      <td>It's primary day in New York and front-runners...</td>
      <td>REAL</td>
    </tr>
  </tbody>
</table>
</div>



We will only use text feature


```python
X = df['text']
print(X)
```

    0       Daniel Greenfield, a Shillman Journalism Fello...
    1       Google Pinterest Digg Linkedin Reddit Stumbleu...
    2       U.S. Secretary of State John F. Kerry said Mon...
    3       — Kaydee King (@KaydeeKing) November 9, 2016 T...
    4       It's primary day in New York and front-runners...
                                  ...                        
    6330    The State Department told the Republican Natio...
    6331    The ‘P’ in PBS Should Stand for ‘Plutocratic’ ...
    6332     Anti-Trump Protesters Are Tools of the Oligar...
    6333    ADDIS ABABA, Ethiopia —President Obama convene...
    6334    Jeb Bush Is Suddenly Attacking Trump. Here's W...
    Name: text, Length: 6335, dtype: object
    


```python
y = df.label
print(y)
```

    0       FAKE
    1       FAKE
    2       REAL
    3       FAKE
    4       REAL
            ... 
    6330    REAL
    6331    FAKE
    6332    FAKE
    6333    REAL
    6334    REAL
    Name: label, Length: 6335, dtype: object
    

# Spliting the Data into Training and Testing Sets


```python
pip install sklearn #installing sklearn library
```

    Collecting sklearn
      Downloading sklearn-0.0.tar.gz (1.1 kB)
    Collecting scikit-learn
      Downloading scikit_learn-0.24.2-cp38-cp38-win_amd64.whl (6.9 MB)
    Collecting scipy>=0.19.1
      Downloading scipy-1.6.3-cp38-cp38-win_amd64.whl (32.7 MB)
    Collecting joblib>=0.11
      Downloading joblib-1.0.1-py3-none-any.whl (303 kB)
    Collecting threadpoolctl>=2.0.0
      Downloading threadpoolctl-2.1.0-py3-none-any.whl (12 kB)
    Requirement already satisfied: numpy>=1.13.3 in c:\users\sldin\appdata\local\programs\python\python38\lib\site-packages (from scikit-learn->sklearn) (1.20.3)
    Using legacy 'setup.py install' for sklearn, since package 'wheel' is not installed.
    Installing collected packages: threadpoolctl, scipy, joblib, scikit-learn, sklearn
        Running setup.py install for sklearn: started
        Running setup.py install for sklearn: finished with status 'done'
    Successfully installed joblib-1.0.1 scikit-learn-0.24.2 scipy-1.6.3 sklearn-0.0 threadpoolctl-2.1.0
    Note: you may need to restart the kernel to use updated packages.
    


```python
from sklearn.model_selection import train_test_split
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 13)
```

# Using TfidfVectorizer to generate stop words

**What is a TfidfVectorizer?**
- **TF (Term Frequency):**

The number of times a word appears in a document is its Term Frequency. 
A higher value means a term appears more often than others, and so, the document is a good match when the term is part of the search terms.

- **IDF (Inverse Document Frequency):**

Words that occur many times a document, but also occur many times in many others, may be irrelevant. 
IDF is a measure of how significant a term is in the entire corpus.

The TfidfVectorizer converts a collection of raw documents into a matrix of TF-IDF features.

Lets initialize the TfidfVectorizer with stop words from the English Language and a maximum document frequency of 0.8.

Stop words are the most common words in a language that are filtered out before processing the natural language data.


```python
from sklearn.feature_extraction.text import TfidfVectorizer
```


```python
TfidfVectorizer?
```


```python
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.8)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
```

**What is a PassiveAggressiveClassifier?**

Passive Aggressive algorithms are online learning algorithms. Such an algorithm remains passive for a correct classification outcome, and turns aggressive in the event of a miscalculation, updating and adjusting. Unlike most other algorithms, it does not converge. Its purpose is to make updates that correct the loss, causing very little change in the norm of the weight vector.

 Initialize the PassiveAggressiveClassifier. We will fit this on tfidf_train and y_train and then predict on the test set from the TfidfVectorizer and calculate the accuracy with accuracy_score() from sklearn.metrics


```python
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
```


```python
PassiveAggressiveClassifier?
```


```python
pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)
```




    PassiveAggressiveClassifier(max_iter=50)



# Predicting on Test Data


```python
y_pred = pac.predict(tfidf_test)
print(y_pred,np.array(y_test))
```

    ['REAL' 'REAL' 'FAKE' ... 'REAL' 'REAL' 'REAL'] ['REAL' 'REAL' 'FAKE' ... 'REAL' 'REAL' 'REAL']
    

# Score


```python
score = accuracy_score(y_test, y_pred)
score = round(score,2)
print("Accuracy: ",score*100)
```

    Accuracy:  95.0
    

Accuracy of 95% with this model. 

# Confusion Matrix


```python
mat = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print("So with this model, we have ",mat[0][0], "true positives, ",mat[1][1], " true negatives, "
      ,mat[1][0], " false positives, and ",mat[0][1], " false negatives.")
```

    So with this model, we have  603 true positives,  595  true negatives,  36  false positives, and  33  false negatives.
    
