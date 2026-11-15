import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("mushrooms.csv")
df = df.replace("?", np.nan)
df = df.dropna()

df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)


def split(df_train, method):
    if len(df_train["class"].unique()) == 1:
        return {"label": df_train["class"].iloc[0]}

    if len(df_train.columns) == 1:
        return {"label": df_train["class"].mode()[0]}

    entropy = 0
    gini = 1
    rows = len(df_train)

    for cls in np.unique(df_train["class"]):
        cnt = (df_train["class"] == cls).sum()
        prob = cnt / rows
        entropy -= prob * math.log2(prob)
        gini -= prob * prob

    wt_avg = 1
    gain_ratio = 0
    mce = 1
    split_feature = None

    for feature in df_train.columns:

        if feature == "class":
            continue

        ig_feat = entropy
        wt_avg_ft = 0
        split_info = 0
        mce_ft = 0

        for ft in np.unique(df_train[feature]):

            entropy_ft = 0
            gini_ft = 1
            max_prob = 0

            copy_df = df_train[df_train[feature] == ft] #child made from values of the current feature
            subset_rows = len(copy_df)

            classes = np.unique(copy_df["class"])

            for cls in classes:
                cnt = (copy_df["class"] == cls).sum()
                prob = cnt / subset_rows

                max_prob = max(max_prob, prob)

                entropy_ft -= prob * math.log2(prob)
                gini_ft -= prob * prob

            ig_feat -= (subset_rows / rows) * entropy_ft
            wt_avg_ft += (subset_rows / rows) * gini_ft

            split_info -= (subset_rows / rows) * math.log2(subset_rows / rows)

            mce_ft += (subset_rows / rows) * (1 - max_prob)

        if method == "gini":
            if wt_avg_ft < wt_avg:
                wt_avg = wt_avg_ft
                split_feature = feature

        if method == "gain ratio":
            if split_info > 0:
                gain_ratio_ft = ig_feat / split_info

                if gain_ratio_ft > gain_ratio:
                    gain_ratio = gain_ratio_ft
                    split_feature = feature

        if method == "misclassification error":
            if mce_ft < mce:
                mce = mce_ft
                split_feature = feature

    if split_feature is None:
        return {"label": df_train["class"].mode()[0]}

    tree = {
        "feature": split_feature,
        "children": {}
    }

    for value in np.unique(df_train[split_feature]):
        subset = df_train[df_train[split_feature] == value]
        subset = subset.drop(columns=split_feature)

        tree["children"][value] = split(subset, method)

    return tree


def print_tree(tree, level=0):
    indent = "    " * level

    if "label" in tree:
        print(indent + "Class =", tree["label"])
        return

    print(indent + "Feature:", tree["feature"])

    for value, subtree in tree["children"].items():
        print(indent + "Value =", value)
        print_tree(subtree, level + 1)


def print_rules(tree, conditions=""):
    if "label" in tree:
        print(f"IF {conditions} THEN class = {tree['label']}")
        return

    feature = tree["feature"]

    for value, subtree in tree["children"].items():
        if conditions == "":
            new_conditions = f"{feature} == {value}"
        else:
            new_conditions = conditions + f" AND {feature} == {value}"

        print_rules(subtree, new_conditions)


tree = split(df_train, "gini")

print("Decision Tree:\n")
print_tree(tree)

print("\nDecision Rules:\n")
print_rules(tree)


def predict(tree, sample):
    if "label" in tree:
        return tree["label"]

    feature = tree["feature"]
    value = sample[feature]

    if value not in tree["children"]:
        return None

    return predict(tree["children"][value], sample)


idx = 823
sample = df_test.iloc[idx]

ans = predict(tree, sample)

print("\nPrediction:", ans)
print("Actual:", sample["class"])


correct = 0

for idx in range(len(df_test)):
    sample = df_test.iloc[idx]
    prediction = predict(tree, sample)

    if prediction == sample["class"]:
        correct += 1

accuracy = correct / len(df_test) * 100

print("\nCustom Decision Tree Accuracy:", accuracy)



X = df.drop(columns=["class"])
y = df["class"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100

print("\nScikit-Learn Decision Tree Accuracy:", accuracy)