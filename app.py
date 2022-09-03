import streamlit as st
import numpy as np
import torch

from prediction import survival,label,LinearNetwork

model=LinearNetwork()
model.load_state_dict(torch.load('patient.pth'))
model.eval()

Ethnicity=['Caucasian','Hispanic','African American','Asian',
       'Native American', 'Other/Unknown']

st.set_page_config(page_title='Patient Survivability Prediction',page_icon='🧬',layout='wide')
st.markdown("<h1 style = 'text-align: center;color: #8A03DD;'> 🧬 Patient Survivability Prediction 🧬 </h1>",unsafe_allow_html=True)
video_html = """
            <center>
            <video controls width="750" autoplay="true" muted="false" loop="true">
<source 
            src="https://media.istockphoto.com/videos/electrocardiogram-loopable-3-in-1-red-video-id450030771" 
            type="video/mp4" />
</video>
</center>

        """
st.markdown(video_html,unsafe_allow_html=True)

def main():
    with st.form('prediction form'):
        st.subheader("Enter the values for the following vitals of the patient: ")

        age=st.text_input('Enter the age of the patient: ')
        ethnicity=st.selectbox('Select the ethnicity of the patient: ',options=Ethnicity)
        glucose_apache=st.text_input('Enter the value of glucose apache of the patient: ')
        creatinine_apache=st.text_input('Enter the value of creatinine apache of the patient: ')
        bun_apache=st.text_input('Enter the value of bun apache of the patient: ')
        apache_2=st.text_input('Enter the value of apache 2 of the patient: ')
        sodium_apache=st.text_input('Enter the value of sodium apache of the patient: ')
        d1_potassium=st.text_input('Enter the value of d1 potassium of the patient: ')
        submit=st.form_submit_button('Predict Patient Survivability')

    if submit:
        age=float(age)
        ethnicity=label(ethnicity,Ethnicity)
        glucose_apache=float(glucose_apache)
        creatinine_apache=float(creatinine_apache)
        bun_apache=float(bun_apache)
        apache_2=float(apache_2)
        sodium_apache=float(sodium_apache)
        d1_potassium=float(d1_potassium)
        inp=np.array([age,ethnicity,glucose_apache,creatinine_apache,bun_apache,apache_2,sodium_apache,d1_potassium]).reshape(1,-1)
        pred=survival(model=model,inp=inp)

        if (pred==0):
            st.write('We are pleased to tell you that the patient will survive.')
        else:
            st.write('We deeply regret to say that the patient is not going to survive.')


if __name__=='__main__':
    main()