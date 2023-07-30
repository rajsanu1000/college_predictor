from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np
import pandas as pd
import joblib


app=Flask(__name__)
@app.route('/')
def hello():
    return render_template('frontpage.html')

@app.route('/cse')
def cse():
    return render_template('predicts.html')

@app.route('/predict',methods=['POST'])
def predict():
    Height=float(request.form.get("Height"))
    model=pickle.load(open('rank1.pkl','rb'))  
    result=model.predict(np.array([Height]).reshape(1,1))
    # print('rank in {}'.format(result))

    df=pd.read_csv("cseIT.csv")

    list1=[]
    list2=[]
    list4=[]
    for i in range(31):
        dfa=df[df.InstituteNo==i+1]
        dfalist=dfa['closing_rank'].values.tolist()
        list1.append(dfalist)

    for i in list1:
        list2.append(max(i))
    dictionary={}
    for i in range(31):
        dictionary[list2[i]]=i+1
    list3=sorted(list2)
    list3.reverse()

    for i in list3:
        list4.append(dictionary[i])
    dict = {1:'NIT-Agartala', 2:'NIT-Allahabad', 3:'NIT-Andhra-Pradesh', 4:'NIT-Arunachal-Pradesh', 5:'NIT-Bhopal', 
        6:'NIT-Calicut', 7:'NIT-Delhi',8:'NIT-Durgapur', 9:'NIT-Goa', 10:'NIT-Hamirpur',
        11:'NIT-Jaipur', 12:'NIT-Jalandhar', 13:'NIT-Jamshedpur', 14:'NIT-Karnataka-Surathkal', 15:'NIT-Kurukshetra',
        16:'NIT-Manipur', 17:'NIT-Meghalaya', 18:'NIT-Mizoram', 19:'NIT-Nagaland', 20:'NIT-Nagpur',
        21:'NIT-Patna', 22:'NIT-Puducherry', 23:'NIT-Raipur', 24:'NIT-Rourkela', 25:'NIT-Sikkim',
        26:'NIT-Silchar', 27:'NIT-Srinagar', 28:'NIT-Surat', 29:'NIT-Tiruchirappalli', 30:'NIT-Uttarakhand', 31:'NIT-Warangal'}

    list5=[]
    k=result


    for i in list4:
        if(i!=k):
            list5.append(dict[i])
        else:
            break
    list5.append(dict[i])
    
    print(list5)
    
    
    return render_template('predicts.html',predict_price='The colleges predicted are - {}'.format(list5))
    

@app.route('/electronics')
def electronics():
    return render_template('predictelec.html')

@app.route('/predictelec',methods=['POST'])
def predictelec():
    Height=float(request.form.get("Height"))
    modelelec=pickle.load(open('electronics.pkl','rb'))  
    result=modelelec.predict(np.array([Height]).reshape(1,1))
    # print('rank in {}'.format(result))

    df=pd.read_csv("Book2.csv")

    list1=[]
    list2=[]
    list4=[]
    for i in range(31):
        dfa=df[df.InstituteNo==i+1]
        dfalist=dfa['closing_rank'].values.tolist()
        list1.append(dfalist)

    for i in list1:
        list2.append(max(i))
    dictionary={}
    for i in range(31):
        dictionary[list2[i]]=i+1
    list3=sorted(list2)
    list3.reverse()

    for i in list3:
        list4.append(dictionary[i])
    dict = {1:'NIT-Agartala', 2:'NIT-Allahabad', 3:'NIT-Andhra-Pradesh', 4:'NIT-Arunachal-Pradesh', 5:'NIT-Bhopal', 
        6:'NIT-Calicut', 7:'NIT-Delhi',8:'NIT-Durgapur', 9:'NIT-Goa', 10:'NIT-Hamirpur',
        11:'NIT-Jaipur', 12:'NIT-Jalandhar', 13:'NIT-Jamshedpur', 14:'NIT-Karnataka-Surathkal', 15:'NIT-Kurukshetra',
        16:'NIT-Manipur', 17:'NIT-Meghalaya', 18:'NIT-Mizoram', 19:'NIT-Nagaland', 20:'NIT-Nagpur',
        21:'NIT-Patna', 22:'NIT-Puducherry', 23:'NIT-Raipur', 24:'NIT-Rourkela', 25:'NIT-Sikkim',
        26:'NIT-Silchar', 27:'NIT-Srinagar', 28:'NIT-Surat', 29:'NIT-Tiruchirappalli', 30:'NIT-Uttarakhand', 31:'NIT-Warangal'}

    list5=[]
    k=result


    for i in list4:
        if(i!=k):
            list5.append(dict[i])
        else:
            break
    list5.append(dict[i])
    
    # print(list5)
    
    
    return render_template('predictelec.html',predict_price='The colleges predicted are - {}'.format(list5))

    
    


@app.route('/mechcivil')
def mechcivil():
    return render_template('mechcivil.html')

@app.route('/predictmechcivil',methods=['POST'])
def predictmechcivil():
    Height=float(request.form.get("Height"))
    model=pickle.load(open('modelmechcivil.pkl','rb'))  
    result=model.predict(np.array([Height]).reshape(1,1))
    # print('rank in {}'.format(result))

    df=pd.read_csv("mechcivil.csv")
    list1=[]
    list2=[]
    list4=[]
    for i in range(31):
        dfa=df[df.InstituteNo==i+1]
        dfalist=dfa['closing_rank'].values.tolist()
        if len(dfalist)==0:
           list1.append([0])
        else:
            list1.append(dfalist)
    for i in list1:
        list2.append(max(i))
    dictionary={}
    for i in range(31):
        dictionary[list2[i]]=i+1
    list3=sorted(list2)
    list3.reverse()

    for i in list3:
        list4.append(dictionary[i])
    dict = {1:'NIT-Agartala', 2:'NIT-Allahabad', 3:'NIT-Andhra-Pradesh', 4:'NIT-Arunachal-Pradesh', 5:'NIT-Bhopal', 
        6:'NIT-Calicut', 7:'NIT-Delhi',8:'NIT-Durgapur', 9:'NIT-Goa', 10:'NIT-Hamirpur',
        11:'NIT-Jaipur', 12:'NIT-Jalandhar', 13:'NIT-Jamshedpur', 14:'NIT-Karnataka-Surathkal', 15:'NIT-Kurukshetra',
        16:'NIT-Manipur', 17:'NIT-Meghalaya', 18:'NIT-Mizoram', 19:'NIT-Nagaland', 20:'NIT-Nagpur',
        21:'NIT-Patna', 22:'NIT-Puducherry', 23:'NIT-Raipur', 24:'NIT-Rourkela', 25:'NIT-Sikkim',
        26:'NIT-Silchar', 27:'NIT-Srinagar', 28:'NIT-Surat', 29:'NIT-Tiruchirappalli', 30:'NIT-Uttarakhand', 31:'NIT-Warangal'}
    list5=[]
    k=result


    for i in list4:
        if(i!=k):
            list5.append(dict[i])
        else:
            break
    list5.append(dict[i])
    
    print(list5)
    
    
    return render_template('mechcivil.html',predict_price='The colleges predicted are - {}'.format(list5))

@app.route('/chemical')
def chemical():
    return render_template('chemical.html')
@app.route('/predictchemical',methods=['POST'])
def predictchemical():
    Height=float(request.form.get("Height"))
    model=pickle.load(open('modelchem.pkl','rb'))  
    result=model.predict(np.array([Height]).reshape(1,1))
    df=pd.read_csv("Chemistryrelated.csv")
    list1=[]
    list2=[]
    list4=[]
    for i in range(31):
        dfa=df[df.InstituteNo==i+1]
        dfalist=dfa['closing_rank'].values.tolist()
        if len(dfalist)==0:
           list1.append([0])
        else:
            list1.append(dfalist)
    for i in list1:
        list2.append(max(i))
    dictionary={}
    for i in range(31):
        dictionary[list2[i]]=i+1
    list3=sorted(list2)
    list3.reverse()
    for i in list3:
        list4.append(dictionary[i])
    dict = {1:'NIT-Agartala', 2:'NIT-Allahabad', 3:'NIT-Andhra-Pradesh', 4:'NIT-Arunachal-Pradesh', 5:'NIT-Bhopal', 
        6:'NIT-Calicut', 7:'NIT-Delhi',8:'NIT-Durgapur', 9:'NIT-Goa', 10:'NIT-Hamirpur',
        11:'NIT-Jaipur', 12:'NIT-Jalandhar', 13:'NIT-Jamshedpur', 14:'NIT-Karnataka-Surathkal', 15:'NIT-Kurukshetra',
        16:'NIT-Manipur', 17:'NIT-Meghalaya', 18:'NIT-Mizoram', 19:'NIT-Nagaland', 20:'NIT-Nagpur',
        21:'NIT-Patna', 22:'NIT-Puducherry', 23:'NIT-Raipur', 24:'NIT-Rourkela', 25:'NIT-Sikkim',
        26:'NIT-Silchar', 27:'NIT-Srinagar', 28:'NIT-Surat', 29:'NIT-Tiruchirappalli', 30:'NIT-Uttarakhand', 31:'NIT-Warangal'}
    list5=[]
    k=result
    for i in list4:
        if(i!=k):
            list5.append(dict[i])
        else:
            break
    list5.append(dict[i])
    print(list5)
    return render_template('chemical.html',predict_price='The colleges predicted are - {}'.format(list5))
if __name__=='__main__':
    app.run(debug=True)
