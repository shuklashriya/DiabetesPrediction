import streamlit as st
import pandas as pd
import polars as pl
import numpy as np
import pickle


st.markdown("<h1 style='text-align: center; color: black;'>Diabetes Prediction</h1>", unsafe_allow_html=True)
model = pickle.load(open('LogisticReg.sav','rb'))
with st.form(key="My form",clear_on_submit=True):
    col1, col2 = st.columns(2,gap="small")
    with col1:
        sex = st.selectbox("Select Gender", options=('Female', 'Male'))
    with col2:
        age=st.selectbox("Age category",options=('18-24','25-29','30-34','35-39','40-44','45-49','50-54','55-59','60-64','65-69','70-74','75-79','80 or older'))
    
    with col1:
        smoker= st.selectbox("Have you smoked at least 100 cigarettes in your entire life? ", options=('No','Yes'))
    with col2:
        hvyAlcoholConsump= st.selectbox("Do You Consume Alcohol? (adult men >=14 drinks per week and adult women>=7 drinks per week)", options=('No','Yes'))

    with col1:
        highBP= st.selectbox("Do You have HIgh BP?", options=('No','Yes'))
    with col2:
        highChol= st.selectbox("Do You have HIgh cholesterol?", options=('No','Yes'))
    
    with col1:
        stroke = st.selectbox("(Ever told) you had a stroke.?", options=('No','Yes'))
    with col2:
        heartDiseaseorAttack= st.selectbox("(Ever told) you had coronary heart disease?", options=('No','Yes'))
    
    physActivity= st.selectbox("Have You Played Any Sports (running, biking, etc.) In The Past Month?", options=('No','Yes'))

    diffWalk=st.selectbox("Do You Face Difficulty Walking Or Climbing Stairs?", options=("No", "Yes"))

    with col1:
        fruits= st.selectbox("Do You Consume Fruit 1 or more times per day ?", options=('No','Yes'))
    with col2:
        veggies= st.selectbox("Do You Consume Vegetables 1 or more times per day?", options=('No','Yes'))
    
    with col1:
        bMICAT=st.selectbox("Select Your BMI category",options=('Under_weight', 'Normal_weight', 'Over_weight', 'Obese'))
    with col2:
        genHlth =st.selectbox("How Can You Define Your General Health?",options=('Very good', 'Fair', 'Good', 'Poor', 'Excellent'))
    
    mentHlth= st.number_input("For How Many Days During The Past 30 Days Your Mental Health Not Good?", 0, 30, 0)
    physHlth= st.number_input("For How Many Days During The Past 30 Days Your Pysical Health Not Good?", 0, 30, 0)
    
    submit = st.form_submit_button("Predict")


if highBP == "NO":
    HighBP = 0
else:
    HighBP = 1


if highChol == "No":
    HighChol = 0
else:
    HighChol = 1

if smoker == "No":
    Smoker = 0
else:
    Smoker = 1

if stroke == "No":
    Stroke = 0
else:
    Stroke = 1

if heartDiseaseorAttack == "No":
    HeartDiseaseorAttack = 0
else:
    HeartDiseaseorAttack = 1

if physActivity == "No":
    PhysActivity = 0
else:
    PhysActivity = 1

if fruits == "No":
    Fruits = 0
else:
    Fruits = 1

if veggies == "No":
    Veggies = 0
else:
    Veggies = 1

if hvyAlcoholConsump == "No":
    HvyAlcoholConsump = 0
else:
    HvyAlcoholConsump = 1

if genHlth == "Poor":
    GenHlth = 5
elif genHlth == "Fair":
    GenHlth = 4
elif genHlth == "Good":
    GenHlth = 3
elif genHlth == "Very good":
    GenHlth = 2
else :
    GenHlth = 1

GenHlth=((GenHlth-1)/(5-1))

PhysHlth= ((physHlth-0)/(30-0))
        
MentHlth= ((mentHlth-0)/(30-0))

if diffWalk == "No":
    DiffWalk = 0
else:
    DiffWalk = 1

if sex == "Female":
    Sex = 0
else:
    Sex = 1 

if age == "18-24":
    Age = 1
elif age == "25-29":
    Age = 2
elif age == "30-34":
    Age = 3
elif age == "35-39":
    Age = 4
elif age == "40-44":
    Age = 5 
elif age == "45-49":
    Age = 6
elif age == "50-54":
    Age = 7
elif age == "55-59":
    Age = 8
elif age == "60-64":
    Age = 9       
elif age == "65-69":
    Age = 10
elif age == "70-74":
    Age = 11 
elif age == "75-79":
    Age = 12
else :
    Age = 13

Age=((Age-1)/(13-1))


if bMICAT == "Under_weight":
    BMICAT = 0
elif bMICAT == "Normal_weight":
    BMICAT = 1
elif bMICAT == "Over_weight":
    BMICAT = 2
else :
    BMICAT = 3  

BMICAT=((BMICAT-0)/(3-0))

variable= [[float(HighBP),float(HighChol),float(Smoker),float(Stroke),float(HeartDiseaseorAttack),float(PhysActivity),float(Fruits),float(Veggies),
            float(HvyAlcoholConsump),float(GenHlth),float(MentHlth),float(PhysHlth),float(DiffWalk),float(Sex),float(Age),float(BMICAT)]]


if submit:
    prediction = model.predict(variable)
    prediction_prob = model.predict_proba(variable)
    if prediction == 0:
        
        st.success(f"**The probability that you will have Diabetes is {round(prediction_prob[0][1] * 100, 2)}%."f" You are healthy!**")
    else:
   
        st.warning(f"**The probability that you will have Diabetes is {round(prediction_prob[0][1] * 100, 2)}%."f" You are Not healthy!**")





