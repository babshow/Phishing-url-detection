import pandas as pd
import numpy as np
import streamlit as st
from keras.models import load_model
from extraObjects import preprocessComment, preprocessBatch
#from keras.models import load_model


@st.cache_data
def load_models():
    """
    Load models
    """
    gru_model = load_model('models/cnn_modelv3 .keras')
    
    return gru_model

gru_model = load_models()

prediction_decoding = {0:"Legitimate", 1:"Phishing"}

def main():
    st.title('Phishing Url Detection')
    html_temp = """
    <div style="background:#051733 ;padding:10px">
    <h2 style="color:white;text-align:center;">Phishing URL Classifier</h2>
    </div>
    """

    st.markdown(html_temp, unsafe_allow_html = True)
    st.divider()

    html_temp2 = """
    <div style="background:#14302f ;padding:10px">
    <p style="color:white;text-align:left;">This project aims to identify a phishing url automatically using deep learning nlp techniques </p>
    <p style="color:white;text-align:left;"><b>Class Codes</b><p>
    <p style="color:white;text-align:left;">0:"Legitimate", 1:"Phishing"</p>
    </div>
    """
    st.markdown(html_temp2, unsafe_allow_html = True)
    st.divider()
    news_story = st.text_area('Enter a Reddit Comment',height=200)

    if st.button('Identify URL'):
        input_data = preprocessComment(news_story)
        gruPred = gru_model.predict(input_data).ravel()
        if gruPred[0] <= 0.5:
            prediction=0 
        else: 
            prediction=1
        

        st.success("The URL provided is a " + prediction_decoding[prediction] + " URL")
    
    st.divider()
    dataset = st.file_uploader("Upload a csv dataset of url links. URLs must be in a 'url' column", type=["csv"], accept_multiple_files = False)

    if st.button('Identify Batch'):
        df = pd.read_csv(dataset)
        comments = df['url']
        input_batch = preprocessBatch(comments)
        gruPred = gru_model.predict(input_batch).ravel()
        predictions=[0 if i <= 0.5 else 1 for i in gruPred ]
        

        st.success("Prediction:" + str(predictions))



if __name__=='__main__': 
    main()