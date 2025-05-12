# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import pickle
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score



# df=pd.read_csv('seattle-weather.csv')
# df['date'] = pd.to_datetime(df['date'])
# df['year'] = df['date'].dt.year
# df['month'] = df['date'].dt.month
# df['day'] = df['date'].dt.day

# X=df[['year', 'month', 'day','precipitation']]
# y=df['weather']


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model=LogisticRegression()
# model.fit(X_train,y_train)
# y_pred=model.predict(X_train)

# print("The DATA COUNT IS:\n",df['weather'].value_counts())
# #.value_counts() -> gives the count of all the tyoes of weathe


# print("Accuracy:", accuracy_score(y_train, y_pred))
# print("Precision (macro):", precision_score(y_train, y_pred, average='macro'))
# print("Recall (macro):", recall_score(y_train, y_pred, average='macro'))
# print("F1 Score (macro):", f1_score(y_train, y_pred, average='macro'))

# print("\nConfusion Matrix:")
# print(confusion_matrix(y_train, y_pred))

# print("\nClassification Report:")
# print(classification_report(y_train, y_pred))




# labels = df['weather'].astype('category').cat.categories
# cm = confusion_matrix(y_train, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')

# # coverting the py file to pickle for deployment
# pickle.dump(model , open('weather.pkl','wb'))
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('seattle-weather.csv')
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

X = df[['year', 'month', 'day', 'precipitation']]
y = df['weather']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
#model = LogisticRegression()
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_train)

print("The DATA COUNT IS:\n", df['weather'].value_counts())

print("Accuracy:", accuracy_score(y_train, y_pred))
print("Precision (macro):", precision_score(y_train, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_train, y_pred, average='macro'))
print("F1 Score (macro):", f1_score(y_train, y_pred, average='macro'))

print("\nConfusion Matrix:")
print(confusion_matrix(y_train, y_pred))

print("\nClassification Report:")
print(classification_report(y_train, y_pred))

labels = df['weather'].astype('category').cat.categories
cm = confusion_matrix(y_train, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Save both model and scaler for use during prediction
pickle.dump((model, scaler), open('weather.pkl', 'wb'))
plt.show()





