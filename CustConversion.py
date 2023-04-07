import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objects as go
import json
import requests
from streamlit_lottie import st_lottie
import pickle
import numpy as np
import xgboost as xgb

url1 = requests.get("https://assets4.lottiefiles.com/private_files/lf30_kwjwqk59.json")
url2 = requests.get("https://assets9.lottiefiles.com/packages/lf20_8SVUiK.json")
url3 = requests.get("https://assets2.lottiefiles.com/packages/lf20_u8jppxsl.json")
url4 = requests.get("https://assets1.lottiefiles.com/packages/lf20_hu3ztirc.json")
url5 = requests.get("https://assets10.lottiefiles.com/packages/lf20_4chtroo0.json") # yes
url6 = requests.get("https://assets4.lottiefiles.com/packages/lf20_qv7z0gx15x.json") # NO

url1_json = url2_json = url3_json = url4_json = url5_json = url6_json = dict()
#url2_json = dict()

if url1.status_code == 200:
    url1_json = url1.json()
if url2.status_code == 200:
    url2_json = url2.json()
if url3.status_code == 200:
    url3_json = url3.json()
if url4.status_code == 200:
    url4_json = url4.json()
if url5.status_code == 200:
    url5_json = url5.json()
if url6.status_code == 200:
    url6_json = url6.json()
else:
    print("Error in the URL")

col_1,col_2,col_3= st.columns(3)
with col_1:
    st_lottie(url1_json)
with col_2:
    st_lottie(url2_json)
#with col_3:
    #st_lottie(url3_json)
with col_3:
    st_lottie(url4_json)

data = pd.read_csv(r'C:\Users\kgeet\OneDrive\Desktop\CustomerConversion.csv')
data = data.drop_duplicates()
data['y_new'] = data['y'].map({'yes' : 1 , 'no' : 0})
data.loc[data.job == 'unknown','job'] = 'blue-collar'
data.loc[data.education_qual == 'unknown','education_qual']='secondary'

#st.title("Customer Conversion Prediction - Data Distribution")
# st.markdown(
#     """
#     <style>
#     .reportview-container {
#         background: url("Pink.jpg")
#     }
#    .sidebar .sidebar-content {
#         background: url("Pink.jpg")
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

with st.sidebar:
    st.header("Data Distribution")
    selected = option_menu(
        menu_title = None,
        options = ['Feature Distribution','Feature Vs Target','Prediction'],
        )
    st_lottie(url3_json)
if selected == 'Feature Distribution':
    st.header('Feature distribution')
    features = ['Age','Job','Call Type','Previous Outcome','Marital Status','Education Qualification',
                'Duration','Number of Calls','Day','Month']
    c = st.selectbox("Select a feature to visualize its distribution",features)
    if c == 'Age':
        fig = px.histogram(data, x= 'age',title = 'Age Distribution')
        fig.update_layout(title_x=0.3)
        st.write(fig)
        data.dur = data.age.clip(0, 71 )
        fig = px.histogram(data, x = 'age',title = ' Age Distribution After Outlier clipping')
        fig.update_layout(title_x=0.3)
        st.write(fig)
    if c == 'Job':
        fig = px.pie(data,names = 'job',title = 'Job Distribution')
        fig.update_traces(textposition = 'outside')
        fig.update_layout(title_x=0.3)
        st.write(fig)
    if c == 'Previous Outcome':
        fig = px.pie(data,names = 'prev_outcome',title = ' Previous Outcome Distribution')
        fig.update_layout(title_x=0.3)
        st.write(fig)
        fig = px.sunburst(data, path = ['prev_outcome','job'],title = 'Previous Outcome and Job Distribution')
        fig.update_layout(title_x=0.3)
        st.write(fig)
    if c == 'Marital Status':
        fig = px.pie(data, names = 'marital', hole = 0.5,title = ' Marital Distribution')
        fig.update_layout(title_x=0.3)
        st.write(fig)
        fig = px.sunburst(data, path = ['prev_outcome','job','marital'],title = 'Previous Outcome, Marital status and Job Distribution')
        st.write(fig)
    if c == 'Education Qualification':
        fig = px.pie(data,names = 'education_qual',title = ' Education Qualification Distribution')
        fig.update_layout(title_x=0.3)
        st.write(fig)
        fig = px.sunburst(data, path = ['prev_outcome','education_qual'],title = 'Previous Outcome and Education Qualification Distribution')
        fig.update_layout(title_x=0.3)
        st.write(fig)
    if c == 'Duration':
        fig = px.histogram(data, x = 'dur',title = ' Duration Distribution')
        fig.update_layout(title_x=0.3)
        st.write(fig)
        data.dur = data.dur.clip(0, 1500 )
        fig = px.histogram(data, x = 'dur',title = ' Duration Distribution After Outlier clipping')
        fig.update_layout(title_x=0.3)
        st.write(fig)
    if c == 'Number of Calls':
        fig = px.histogram(data, x = 'num_calls',title = ' NUmber of calls Distribution')
        fig.update_layout(title_x=0.3)
        st.write(fig)
        data.num_calls = data.num_calls.clip(0, 14 )
        fig = px.histogram(data, x = 'num_calls',title = ' Number of calls Distribution After Outlier clipping')
        fig.update_layout(title_x=0.3)
        st.write(fig)
    if c == 'Day':
        fig = px.histogram(data, x= 'day',title = ' Contact Day of the month Distribution')
        fig.update_layout(title_x=0.3)
        st.write(fig)
        fig = px.histogram(data, x = 'day',y='num_calls',title = ' Day vs Number of calls Distribution')
        fig.update_layout(title_x=0.3)
        st.write(fig)
    if c == 'Month':
        fig = go.Figure(go.Pie(labels = data['mon'],pull = [0.5,0,0,0,0,0,0,0,0,0,0,0]))
        fig.update_traces(textposition = 'outside')
        st.write(fig)
        fig = px.histogram(data, x = 'mon',y='num_calls',title = ' Month vs Number of calls Distribution')
        fig.update_layout(title_x=0.3)
        st.write(fig)
if selected == 'Feature Vs Target':
    st.header('Feature Vs Target Distribution')
    features_target = ['Age','Job','Call Type','Previous Outcome','Marital Status','Education Qualification',
                        'Duration','Number of Calls','Day','Month']
    t = st.selectbox("Select a feature for its distribution", features_target )
    if t == 'Call Type':
        fig = px.sunburst(data, path = ['call_type','y'],title = 'Call_Type Vs Target')
        fig.update_layout(title_x=0.5)
        st.write(fig)
        call_type = data.groupby('call_type')['y_new'].mean().sort_values(ascending = False)
        fig = px.bar(call_type, title = 'Call_type vs target')
        st.write(fig)
    if t == 'Age':
        fig = px.histogram(data, x= 'age', y = 'y_new',title = 'Age vs Target')
        fig.update_layout(title_x=0.3)
        st.write(fig)
    if t == 'Job':
        fig = px.sunburst(data, path = ['job','y'],title = 'Job Vs Target')
        fig.update_layout(title_x=0.3)
        st.write(fig)
        fig = px.bar(data.groupby('job')['y_new'].mean().sort_values(ascending = False))
        st.write(fig)
    if t == 'Previous Outcome':
        fig = px.sunburst(data, path = ['prev_outcome','y'],title = 'Previous Outcome Vs Target')
        fig.update_layout(title_x=0.3)
        st.write(fig)
        fig = px.bar(data.groupby('prev_outcome')['y_new'].mean().sort_values(ascending = False))
        st.write(fig)
        fig = px.sunburst(data, path = ['job','prev_outcome','y'],title = 'Job, Previous Outcome Vs Target')
        fig.update_layout(title_x=0.3)
        st.write(fig)
    if t == 'Marital Status':
        fig = px.sunburst(data, path = ['marital','y'],title = 'Marital Vs Target')
        fig.update_layout(title_x=0.3)
        st.write(fig)
        fig = px.bar(data.groupby('marital')['y_new'].mean().sort_values(ascending = False))
        st.write(fig)
        fig = px.sunburst(data, path = ['marital','prev_outcome','y'],title = 'Marital, Previous Outcome Vs Target')
        fig.update_layout(title_x=0.3) 
        st.write(fig)
    if t == 'Education Qualification':
        fig = px.sunburst(data, path = ['education_qual','y'],title = 'Education Qualification Vs Target')
        fig.update_layout(title_x=0.3)
        st.write(fig)
        fig = px.bar(data.groupby('education_qual')['y_new'].mean().sort_values(ascending = False))
        st.write(fig)
        fig = px.sunburst(data, path = ['education_qual','prev_outcome','y'],title = 'Education Qualification and Previous Outcome Vs Target')
        fig.update_layout(title_x=0.3)
        st.write(fig)
    if t == 'Month':
        fig = px.sunburst(data, path = ['mon','y'],title = 'Month Vs Target')
        fig.update_layout(title_x=0.3)
        st.write(fig)
        fig = px.bar(data.groupby('mon')['y_new'].mean().sort_values(ascending = False))
        st.write(fig)
        fig = px.sunburst(data, path = ['mon','prev_outcome','y'],title = 'Education Qualification and Previous Outcome Vs Target')
        fig.update_layout(title_x=0.3)
        st.write(fig)
    if t == 'Day':
        fig = px.scatter_3d(data, x = 'num_calls' , y = 'age' , z= 'y',color = 'y',title = 'Number of calls and Age Vs Target Dist')
        st.write(fig)
        fig = px.scatter_3d(data, x = 'num_calls' , y = 'age' , z= 'mon', color = 'y',title = 'Number of calls , Age and Month Vs Target Dist')
        st.write(fig)
    if t == 'Duration':
        fig = px.scatter(data, x = 'dur' , y = 'y' , color = 'y',title = 'Number of calls and Age Vs Target Dist')
        st.write(fig)
        fig = px.scatter_3d(data, x = 'num_calls' , y = 'dur' , z= 'mon', color = 'y',title = 'Number of calls , Duration and Month Vs Target Dist')
        st.write(fig)

if selected == 'Prediction':
    loaded_model = pickle.load(open("trained_model.pkl",'rb'))

    def conversion_prediction(input):
 
        prediction = loaded_model.predict(input)    
        return prediction
   

    def main():
        
        st.header('Enter the details of the customer for the conversion prediction')
        age = st.text_input('Age')
        
        job_list = ['Admin','Blue collar','Entrepreneur','House maid','Management','Retired','Self_employed',
                    'Service','Student','Tech','UnEmployed']
        job_cat= st.selectbox('Click for the Job options',job_list)
        if job_cat == 'Admin':
            job = 0
        if job_cat == 'Blue collar':
            job =  1
        if job_cat == 'Entrepreneur':
            job = 2
        if job_cat == 'House maid':
            job = 3
        if job_cat == 'Management':
            job = 4
        if job_cat == 'Retired':
            job = 5
        if job_cat == 'Self_employed':
            job = 6 
        if job_cat == 'Service':
            job = 7
        if job_cat =='Student':
            job = 8
        if job_cat == 'Tech':
            job = 9
        if job_cat == 'UnEmployed':
            job = 10
        
        marital_list = ['Divorced','Married','Single']
        marital_cat= st.selectbox('Click for the options',marital_list)
        if marital_cat ==  'Divorced':
            marital = 0
        if marital_cat == 'Married':
            marital = 1
        if marital_cat == 'Single':
            marital = 2
        
        education_list = ['Primary','Secondary','Teritary']
        edu_cat = st.selectbox('Click for the options',education_list)
        if edu_cat == 'Primary':
            education_qual = 0
        if edu_cat == 'Secondary':
            education_qual = 1
        if edu_cat == 'Teritary':
            education_qual = 2

        call_list = ['Cellular','Telephone','Unknown']
        call_cat =	st.selectbox('By which mode the call was placed?',call_list)
        if call_cat == 'Cellular':
            call_type = 0
        if call_cat == 'Telephone':
            call_type = 1
        if call_cat == 'Unknown':
            call_type = 2
        
        day = st.text_input('Day of the month')

        mon_list = ['January','February','March','April','May','June','July','August','September','October','November','December']
        mon_cat = st.selectbox("The call is placed in the month of",mon_list)
        if mon_cat == 'January':
            mon = 4
        if mon_cat == 'February':
            mon = 3
        if mon_cat == 'March':
            mon = 7
        if mon_cat == 'April':
            mon = 0
        if mon_cat == 'May':
            mon = 8
        if mon_cat == 'June':
            mon = 6
        if mon_cat == 'July':
            mon = 5
        if mon_cat == 'August':
            mon = 1
        if mon_cat == 'September':
            mon = 11
        if mon_cat == 'October':
            mon = 10
        if mon_cat == 'November':
            mon = 9
        if mon_cat == 'December':
            mon = 2
            
        dur=st.text_input('Duration of the call')

        num_calls= st.text_input('Number of calls to the customer')

        prev_list = ['Failure','Other','Success','Unknown']
        prev_cat = st.selectbox("Mention the outcome of previous call?",prev_list)
        if prev_cat == 'Failure':
            prev_outcome = 0
        if prev_cat == 'Other':
            prev_outcome = 1
        if prev_cat == 'Success':
            prev_outcome = 2
        if prev_cat == 'Unknown':
            prev_outcome = 3
        

        conversion = ''

        if st.button('Predict'):

            input_col = ['age', 'job', 'marital', 'education_qual', 'call_type', 'day', 'mon', 'dur', 'num_calls', 'prev_outcome']
            input_data_1 = [age, job, marital, education_qual, call_type, day, mon, dur, num_calls, prev_outcome]
            input_data = np.array(input_data_1, dtype=int)
            
            input = pd.DataFrame(np.column_stack(input_data),columns = input_col)

            conversion = conversion_prediction(input) #

            if conversion == 0:
                st.error('The contacted cutomer will not buy an insurance')
            else:
                st.success(' The contacted customer will buy an insurance')
            col_4,col_5,col_6= st.columns(3)
            with col_5:
                if conversion == 1:
                    st_lottie(url5_json) # tick
                else:  
                    st_lottie(url6_json) # NO       

    if __name__ =='__main__':
        main()

