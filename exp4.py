import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split  
from pprint import pprint 
from sklearn.preprocessing import LabelEncoder 
from sklearn.naive_bayes import CategoricalNB 
from sklearn.metrics import confusion_matrix 
 
df=pd.read_csv("mushrooms.csv") 
df.replace('?',np.nan,inplace=True) 
 
mode_e=df[df['class']=='e'].mode().iloc[0] 
mode_p=df[df['class']=='p'].mode().iloc[0] 
 
df.update(df[df['class'] == 'e'].fillna(mode_e)) 
df.update(df[df['class'] == 'p'].fillna(mode_p)) 
 
df.drop_duplicates(inplace=True) 
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42) 
 
prior_edible=len(train_set[train_set['class']=='e'])/len(train_set['class']) 
prior_poisonous=len(train_set[train_set['class']=='p'])/len(train_set['class']) 
 
features=['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 
       'stalk-surface-below-ring', 'stalk-color-above-ring', 
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 
       'ring-type', 'spore-print-color', 'population', 'habitat'] 
 
def given(dataset,target_class): 
    given_prob={} 
    class_data=dataset[dataset['class']==target_class] 
    class_count=len(class_data) 
    
    for col in features: 
        given_prob[col]={} 
        
        for value in dataset[col].unique(): 
            count=(class_data[col]==value).sum() 
            given_prob[col][value]=count/class_count 
    
    return given_prob 
 
 
prob_edible = given(train_set,'e') 
prob_poisonous = given(train_set,'p') 
 
 
def predict_single(row): 
    score_edible = np.log(prior_edible) 
    score_poisonous = np.log(prior_poisonous) 
     
    for col in features: 
        value = row[col] 
         
        prob_e = prob_edible[col].get(value, 0) 
        prob_p = prob_poisonous[col].get(value, 0) 
         
        if prob_e > 0: 
            score_edible += np.log(prob_e) 
        else: 
            score_edible += np.log(1e-10) 
             
        if prob_p > 0: 
            score_poisonous += np.log(prob_p) 
        else: 
            score_poisonous += np.log(1e-10) 
            
        print(f"edible: {np.exp(score_edible)} poisonous: {np.exp(score_poisonous)}\n") 
        
    return 'e' if score_edible > score_poisonous else 'p' 
 
def evaluate(test_data): 
    predictions = [] 
    accuracy=0 
    
    for _,row in test_data.iterrows(): 
        prediction = predict_single(row) 
        predictions.append(prediction) 
        
        if prediction==row['class']: 
            accuracy+=1 
            
    accuracy=(accuracy/len(test_data['class'])) 
    return accuracy, predictions 
 
accuracy, y_pred = evaluate(test_set) 

print("Precomputed Conditional Probablity of all columns given edible:\n") 
pprint(prob_edible) 

print("\nPrecomputed Conditional Probablity of all columns given poisonous:\n") 
pprint(prob_poisonous) 
 
df_encoded = df.copy() 

for col in df_encoded.columns: 
    le = LabelEncoder() 
    df_encoded[col] = le.fit_transform(df_encoded[col]) 
 
X = df_encoded[features] 
y = df_encoded['class'] 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
) 
 
sk_model = CategoricalNB() 
sk_model.fit(X_train, y_train) 

sk_predictions = sk_model.predict(X_test) 
sk_accuracy = np.mean(sk_predictions == y_test) 
 
print(f"Test Accuracy(manual): {accuracy * 100:.2f}%") 
print(f"\nScikit-Learn Model Accuracy: {sk_accuracy * 100:.2f}%") 
 
print("\nManual Model Confusion Matrix:") 
print(confusion_matrix(test_set['class'], y_pred, labels=['e', 'p'])) 
 
print("\nScikit-Learn Confusion Matrix:") 
print(confusion_matrix(y_test, sk_predictions, labels=[0, 1]))