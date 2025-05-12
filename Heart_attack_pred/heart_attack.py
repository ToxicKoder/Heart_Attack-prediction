import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import seaborn as sns
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score


df=pd.read_csv("Medicaldataset.csv")
X = df[['Age','Heart rate','Blood sugar']]
y = df['Result']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_train)

print("The shape is",df.shape)

print("Accuracy:", accuracy_score(y_train, y_pred))
print("Precision (macro):", precision_score(y_train, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_train, y_pred, average='macro'))
print("F1 Score (macro):", f1_score(y_train, y_pred, average='macro'))

print("\nConfusion Matrix:")
print(confusion_matrix(y_train, y_pred))

print("\nClassification Report:")
print(classification_report(y_train, y_pred))

labels = df['Result'].astype('category').cat.categories
cm = confusion_matrix(y_train, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

pickle.dump((model, scaler), open('heart_attack.pkl', 'wb'))
plt.show()
















# df['result_numeric'] = df['Result'].map({'negative': 0, 'positive': 1})
# features = ['Age', 'Gender', 'Heart rate', 'Blood sugar']
# colors = df['result_numeric'].map({0: 'blue', 1: 'orange'})

# plt.figure(figsize=(14, 10))

# for i, feature in enumerate(features, 1):
#     plt.subplot(2, 2, i)
#     plt.scatter(df[feature], df['result_numeric'], c=colors, alpha=0.7)
#     plt.xlabel(feature)
#     plt.ylabel('Result (0=negative, 1=positive)')
#     plt.title(f'{feature} vs Result')

# legend_elements = [
#     Line2D([0], [0], marker='o', color='w', label='Negative', markerfacecolor='blue', markersize=8),
#     Line2D([0], [0], marker='o', color='w', label='Positive', markerfacecolor='orange', markersize=8)
# ]

# plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)
# plt.tight_layout()
# plt.show()



# from sklearn.ensemble import RandomForestClassifier
# import pandas as pd
# import matplotlib.pyplot as plt

# X = df[['Age', 'Gender', 'Heart rate', 'Blood sugar']]
# y = df['result_numeric']




# model = RandomForestClassifier(random_state=42)
# model.fit(X, y)

# importances = model.feature_importances_

# for feature, importance in zip(X.columns, importances):
#     print(f"{feature}: {importance:.4f}")

# plt.barh(X.columns, importances)
# plt.xlabel("Feature Importance")
# plt.ylabel("Feature")
# plt.title("Feature Importance (Random Forest)")
# plt.show()


