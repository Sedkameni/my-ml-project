####################################################3
## Part 1 - Setup a Simple Machine Learning Model
#####################################################333

## 1 - Open a terminal or code editor.
## 2 - Create a new Python file named model.py.
## 3 - Copy the following code to train a basic decision tree classifier and save it as model.pkl
# model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def train_and_save():

    # Load the dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Save the trained model
    with open("model.pkl", "wb") as f:
        pickle.dump(clf, f)

    print("Model trained and saved as model.pkl")