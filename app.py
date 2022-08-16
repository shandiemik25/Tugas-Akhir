import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle
df = pd.read_csv('dataset2.csv')
for col in ['label']:
    df[col] = df[col].astype('category')
df = df.sample(frac=1)
x = df[["R", "G", "B"]]
y = df["label"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
# Fit model on training set
model = GradientBoostingClassifier(random_state=0)
model.fit(x_train,y_train)
pred_rfc = model.predict(x_test)
# Save the model to disk
filename = 'model.pkl'
pickle.dump(model, open(filename, 'wb'))

# ...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)