import pandas as pd
import numpy as np
from flask import *
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
app=Flask(__name__)
pickle_in=open('model.pkl','rb')
classifier=pickle.load(pickle_in)
# df=pd.read_csv(r'E:\Work_Data\titanic.csv')
# X = df[['Age', 'Fare', 'Sex', 'sibsp', 'Pclass']]
# y = df[['2urvived']]
print('------------------')
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=53)
# X_train.isnull().sum()

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/model')
def model():
    # print('------------------')
    # df=pd.read_csv(r'E:\Work_Data\titanic.csv')
    #
    # print('$$$$$$$$$$$$$$$$$$$')
    # X_train.dropna(axis=0,inplace=True)
    # X_test.dropna(axis=0,inplace=True)
    # model = RandomForestClassifier()
    # model.fit(X_train, y_train)
    # y_prd = model.predict(X_test)
    # dd=accuracy_score(y_prd, y_test)
    # print(dd)
    return render_template('services.html')
@app.route('/prediction',methods=['POST','GET'])
def prediction():
    print('ssssss')
    if request.method == 'POST':
        print('mmmm')
        f1=request.form['f1']
        f2=request.form['f2']
        f3=request.form['f3']
        f4=request.form['f4']
        f5=request.form['f5']
        print('kkkk')
        val=[int(f1),int(f2),int(f3),int(f4),int(f5)]
        model=RandomForestClassifier()
        # model.fit(X_train,y_train)
        # filename = 'finalized_model.sav'
        # pickle.dump(model, open(filename, 'wb'))
        # print('llll')

        # filename = 'finalized_model.sav'
        # pickle_in = open('model.pkl', 'rb')
        # loaded_model = pickle.load(open(filename, 'rb'))
        result = classifier.predict([val])
        print('llll')
        # result=model.predict([val])
        print(result)
        if result==0:
            a='Survived'
        else:
            a='Not Survived'
        return render_template('portfolio.html',result=result)
    return render_template('portfolio.html')


if __name__=='__main__':
    app.run(host='0.0.0.0', port = 8000)

