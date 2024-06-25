import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import xgboost as xgb
import time


# list of CSV files to read
file_list = ['VGG-16_Aug.csv','Resnet-101_Aug.csv']

classifiers = dict()
for file in file_list:
    classifiers[file] =[
        {
            'name': 'SVM',
            'classifier': SVC(kernel='rbf', C=.01, gamma='scale', class_weight='balanced')
        },

        {
            'name': 'Random Forest',
            'classifier': RandomForestClassifier(n_estimators=100, max_depth=100, random_state=42)
        },

        {
            'name': 'Decision Tree',
            'classifier': DecisionTreeClassifier(criterion='entropy', splitter='random', 
                                                  max_depth=100, min_samples_split=2, 
                                                  min_samples_leaf=1, random_state=42)
        },
                
        {
            'name': 'XGB Classifier',
            'classifier': xgb.XGBClassifier(max_depth=100,learning_rate=.1,n_estimators=100,
                                            subsample=0.001,reg_alpha=.1)
        }
    ]

start_time = time.time()

# loop through each file
for file in file_list:
    # read CSV file into a pandas DataFrame
    df = pd.read_csv(file)
    
    # Create a mapping dictionary for labels
    label_map = {"no": 0, "yes": 1}

    # Replace string labels with numerical labels
    df['Label'] = df['Label'].replace(label_map)
    
    # split data into features and labels
    X = df.drop('Label', axis=1)
    y = df['Label']
    
    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # fit and predict using each classifier for the current file
    print(f"Results for {file}:")
    print("=======================================")
    for clf in classifiers[file]:
        classifier = clf['classifier']
        name = clf['name']
        
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        confmtrx=confusion_matrix(y_test, y_pred)
        
        # print accuracy for current classifier and file
        print(f"Accuracy for {name}: {accuracy}")
        print(f"precision for {name}: {precision}")
        print(f"recall for {name}: {recall}")
        print(f"f1_score for {name}: {f1}")
        print(f"conf_mtrx for {name}+\n: {confmtrx}")
        print('----------------------------------------------')
    
    print("\n")
    end_time = time.time()

    elapsed_time = end_time - start_time
    print("DS2 with select Elapsed time:", elapsed_time, "seconds")