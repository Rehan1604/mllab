import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv("mushrooms.csv")
df = df.replace("?", np.nan)
df = df.dropna()

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

def entropy(col):
    e = 0
    for cnt in col.value_counts():
        p = cnt / len(col)
        e -= p * math.log2(p)
    return e

def information_gain(df_train, feature):

    total_entropy = entropy(df_train["class"])

    for value in df_train[feature].unique():

        subset = df_train[df_train[feature] == value]

        total_entropy -= (len(subset) / len(df_train)) * entropy(subset["class"])

    return total_entropy

def split(df_train):

    if len(df_train["class"].unique()) == 1:
        return {
            "label": df_train["class"].iloc[0]
        }

    if len(df_train.columns) == 1:
        return {
            "label": df_train["class"].mode()[0]
        }

    best_feature = None
    best_gain = -1

    print("\nInformation Gain")

    for feature in df_train.columns:

        if feature == "class":
            continue

        ig = information_gain(df_train, feature)

        print(feature, "=", round(ig, 5))

        if ig > best_gain:
            best_gain = ig
            best_feature = feature

    print("Best Feature:", best_feature)

    tree = {
        "feature": best_feature,
        "children": {}
    }

    for value in df_train[best_feature].unique():

        subset = df_train[df_train[best_feature] == value]

        subset = subset.drop(columns=best_feature)

        tree["children"][value] = split(subset)

    return tree

tree = split(df_train)

print("\nDecision Tree")
print(tree)

def predict(tree, sample):

    if "label" in tree:
        return tree["label"]

    feature = tree["feature"]

    value = sample[feature]

    if value not in tree["children"]:
        return None

    return predict(tree["children"][value], sample)

correct = 0

for idx in range(len(df_test)):

    sample = df_test.iloc[idx]

    prediction = predict(tree, sample)

    if prediction == sample["class"]:
        correct += 1

accuracy = correct / len(df_test) * 100

print("\nID3 Accuracy =", round(accuracy, 2), "%")

print("\nRules")

def rules(tree, path=""):

    if "label" in tree:
        print(path + " -> " + tree["label"])
        return

    feature = tree["feature"]

    for value in tree["children"]:

        rules(tree["children"][value],
              path + feature + "=" + str(value) + " ")

rules(tree)

df2 = pd.read_csv("mushrooms.csv")

for col in df2.columns:
    df2[col] = LabelEncoder().fit_transform(df2[col])

X = df2.drop(columns=["class"])
y = df2["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100

print("\nSklearn Accuracy =", round(accuracy, 2), "%")