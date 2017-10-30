from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import urllib.parse
import os

def loadFile(name):
    directory = str(os.getcwd())
    filepath = os.path.join(directory, name)
    with open(filepath,'r', encoding='UTF8') as f:
        data = f.readlines()
    data = list(set(data))
    result = []
    for d in data:
        d = str(urllib.parse.unquote(d))   #converting url encoded data to simple string
        result.append(d)
    return result

harmful_requests = loadFile('harmful_requests.txt')
valid_requests = loadFile('valid_requests.txt')

harmful_requests = list(set(harmful_requests))
valid_requests = list(set(valid_requests))
allQueries = harmful_requests + valid_requests
yBad = [1 for i in range(0, len(harmful_requests))]  #labels, 1 for malicious and 0 for clean
yGood = [0 for i in range(0, len(valid_requests))]
y = yBad + yGood
queries = allQueries

#converting data to vectors
# The tf-idf value increases proportionally to the number of times a word appears in the document
vectorizer = TfidfVectorizer(min_df = 0.0, analyzer="char", sublinear_tf=True, ngram_range=(1,3))
X = vectorizer.fit_transform(queries)

#splitting dataset into training and testing samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#counting length of dataset
harmfulCount = len(harmful_requests)
validCount = len(valid_requests)

lgs = LogisticRegression(class_weight={1: 2 * validCount / harmfulCount, 0: 1.0}) # class_weight='balanced')
lgs.fit(X_train, y_train)

predicted = lgs.predict(X_test)

fpr, tpr, _ = metrics.roc_curve(y_test, (lgs.predict_proba(X_test)[:, 1]))
auc = metrics.auc(fpr, tpr)

print("No of harmful requests: %d" % harmfulCount)
print("No of valid requsts: %d" % validCount)
print("Baseline Constant negative: %.6f" % (validCount / (validCount + harmfulCount)))
print("------------")
print("Accuracy: %f" % lgs.score(X_test, y_test))  #checking the accuracy
print("Precision: %f" % metrics.precision_score(y_test, predicted))
print("Recall: %f" % metrics.recall_score(y_test, predicted))
print("F1-Score: %f" % metrics.f1_score(y_test, predicted))
print("AUC: %f" % auc)
