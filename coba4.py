import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv('dataset2.csv')
df

for col in ['label']:
    df[col] = df[col].astype('category')
df.dtypes

df = df.sample(frac=1)

x = df[["R", "G", "B"]]
y = df["label"]




x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

from sklearn.ensemble import GradientBoostingClassifier
GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(x_train, y_train)

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(x_train,y_train)

from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(random_state=42)
RFC.fit(x_train, y_train)

pred_rfc = RFC.predict(x_test)


RFC.score(x_test, y_test)

print(classification_report(y_test, pred_rfc))

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("esp32andfirebase-f0541-firebase-adminsdk-5x1rd-3dcc708696.json")
firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://esp32andfirebase-f0541-default-rtdb.asia-southeast1.firebasedatabase.app/'
        })

if not firebase_admin._apps:
    default_app = firebase_admin.initialize_app(cred)

ref = db.reference('/Sensor')
ref_r = db.reference('/Sensor/r')
ref_g = db.reference('/Sensor/g')
ref_b = db.reference('/Sensor/b')

print(ref.get())

def output_lable(n):
    if n == 1:
        return "Uang Seribu Rupiah"
    elif n == 2:
        return "Uang Dua Ribu Rupiah"
    elif n == 5:
        return "Uang Lima Ribu Rupiah"
    elif n == 10:
        return "Uang Sepuluh Ribu Rupiah"
    elif n == 50:
        return "Uang Lima Puluh Ribu Rupiah"
    else :
        return "Uang Seratus Ribu Rupiah"

ref_new = db.reference('Output')
nom_ref_new = ref_new.child('Output_nominal')

def manual_testing(r, g, b):
    new_def_test = pd.DataFrame([[r,g,b]], columns=['R', 'G', 'B'])
    pred_GBC = GBC.predict(new_def_test)
    output = str(output_lable(pred_GBC[0]))
    ref_new.update({
        'Output_nominal': output
    })
    print(output)

manual_testing(ref_r.get(), ref_g.get(), ref_b.get())