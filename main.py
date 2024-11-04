import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, max_error, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    pd.options.mode.chained_assignment = None
    test = pd.read_csv(r'C:\Users\utsav\Downloads\archive\UNSW_NB15_testing-set.csv')  # Reading csv file for Test data
    train = pd.read_csv(r'C:\Users\utsav\Downloads\archive\UNSW_NB15_training-set.csv')  # Reading csv file for Train data
    data = pd.concat([test, train])  # Concatenating test and train data
    data = data.drop(columns=['id', 'proto', 'state', 'service', 'label'])  # Remove useless attributes
    y = data['attack_cat']  # Target Variable
    data = data.drop('attack_cat', axis=1)  # Remaining dataset
    xtrain, xtest, ytrain, ytest = train_test_split(data, y, test_size=0.3, random_state=101)  # Train-test Splitting
    xtrain, xtest = feature_extraction(xtrain, ytrain, xtest, data)  # Feature Extraction
    
    print("For Train Data: \n")
    ytrain = preprocessing(ytrain)  # Preprocessing target train and test data
    print("For Test Data: \n")
    ytest = preprocessing(ytest)
    classification(xtrain, xtest, ytrain, ytest)

def feature_extraction(xtrain, ytrain, xtest, data):  # Feature Extraction using ExtraTreeClassifier
    ETC = ExtraTreesClassifier()
    ETC.fit(xtrain, ytrain)
    m = np.mean(ETC.feature_importances_)
    for i in range(len(ETC.feature_importances_)):
        if ETC.feature_importances_[i] < m:
            xtrain = xtrain.drop(labels=data.columns[i], axis=1)
            xtest = xtest.drop(labels=data.columns[i], axis=1)
    return xtrain, xtest

def preprocessing(data):  # Encoding
    unique_vals = list(data.unique())
    mapping_dict = {val: i for i, val in enumerate(unique_vals)}
    data = data.map(mapping_dict)
    data = data.astype('int')  # Changing data type of y
    return data

def classification(xtrain, xtest, ytrain, ytest):
    classifiers = [GaussianNB(), DecisionTreeClassifier(), RandomForestClassifier(), KNeighborsClassifier()]
    classifier_names = ["Gaussian NB", "Decision Tree", "Random Forest", "KNN"]
    
    accuracy_scores, f1_scores, precision_scores, recall_scores, max_errors, times = [], [], [], [], [], []
    
    for i, clf in enumerate(classifiers):
        print(f"\n{classifier_names[i]}:")
        ac, f1, ps, rs, me, t = classify(xtrain, xtest, ytrain, ytest, clf)
        accuracy_scores.append(ac)
        f1_scores.append(f1)
        precision_scores.append(ps)
        recall_scores.append(rs)
        max_errors.append(me)
        times.append(t)
    
    result(classifier_names, accuracy_scores, f1_scores, precision_scores, recall_scores, max_errors, times)

def classify(xtrain, xtest, ytrain, ytest, clf):
    start_time = time.time()
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)
    
    accuracy = accuracy_score(ytest, ypred)
    f1 = f1_score(ytest, ypred, labels=np.unique(ypred), average='weighted', zero_division=1)
    precision = precision_score(ytest, ypred, labels=np.unique(ypred), average='weighted', zero_division=1)
    recall = recall_score(ytest, ypred, labels=np.unique(ypred), average='weighted', zero_division=1)
    max_err = max_error(ytest, ypred)
    elapsed_time = time.time() - start_time
    
    print(f"Accuracy Score for Classifier = {accuracy:.2f}")
    print(f"F1 Score for Classifier = {f1:.2f}")
    print(f"Precision Score for Classifier = {precision:.2f}")
    print(f"Recall Score for Classifier = {recall:.2f}")
    print(f"Max Error for Classifier = {max_err:.2f}")
    print(f"Classification Time = {elapsed_time:.2f}")
    print("Confusion Matrix:")
    plot_confusion_matrix(ytest, ypred)
    
    return accuracy, f1, precision, recall, max_err, elapsed_time

def plot_confusion_matrix(y_test, y_pred):
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def result(classifier_names, accuracy_scores, f1_scores, precision_scores, recall_scores, max_errors, times):
    plt.figure(figsize=(20, 10))
    
    metrics = [
        ("Accuracy Score", accuracy_scores, 1),
        ("F1 Score", f1_scores, 2),
        ("Precision Score", precision_scores, 3),
        ("Recall Score", recall_scores, 4),
        ("Max Error", max_errors, 5),
        ("Time", times, 6)
    ]
    
    for title, metric, subplot_num in metrics:
        plt.subplot(2, 3, subplot_num)
        plt.plot(classifier_names, metric, label=title, marker='o', markersize=8)
        plt.xlabel("Classifier")
        plt.ylabel("Metrics")
        plt.title(f"{title} Comparison")
        plt.legend()
    
    plt.tight_layout()
    plt.show()

if _name_ == "_main_":
    main()