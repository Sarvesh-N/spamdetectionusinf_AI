import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
from PIL import Image

image = Image.open('test1.jpeg')

ps = PorterStemmer()

st.markdown("""<style>
            .css-15zrgzn.eqr7zpz2{
            visibility : hidden;
            }
            .css-cio0dv.e1g8pov61{
            visibility : hidden;
            }
            #email-sms-spam-classifier{
            
            margin-right : 200px;
            } 
            
            .css-10trblm.eqr7zpz0{
            display : flux;
            align-items: left;
            }
            
            </style>""", unsafe_allow_html=True)

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.markdown("## Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")
text = st.empty()
text.write("#### Waiting for your message...")
if st.button('Predict'):

    if input_sms != "":
        # 1. preprocess
        transformed_sms = transform_text(input_sms)
        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])
        # 3. predict
        result = model.predict(vector_input)[0]
        print(result)
        # 4. Display
        if result == 1:
            text.write("#### Spam!")
        else:
            text.write("#### Not Spam")

    else:
        st.write("#### Enter any message and try again")



# Add some custom CSS to style the header
st.markdown(
    """
    <style>
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
    }
    .header-image {
        width: 50px;  /* Adjust the width as needed */
        margin-right: 1rem;
    }
    .css-1v0mbdj.ebxwdo61{
    display:flux;
    position: relative;
    right: 300px;
    bottom : 400px;
    }
    body {
     background-image: url("test1.jpeg");
     background-size:cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)







