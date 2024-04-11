import streamlit as st
import pickle
import re
import nltk
from joblib import load

nltk.download('punkt')
nltk.download('stopwords')

#loading models
clf = load('knnClassifier.joblib')
tfidfd = pickle.load(open('vec.pkl','rb'))

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text
# web app

st.set_page_config("wide")


with st.sidebar:
    st.header(":blue[About Me] :man:")
    st.write("I am an AI and Data Science Student. Passionate about Data science and ML")
    github_emoji = "\U0001F680"
    github_link = f"[Github Profile {github_emoji}](https://github.com/BHEESETTIANAND)"
    st.markdown(github_link, unsafe_allow_html=True)
    st.write("To see my work, please visit the link to my portfolio below.")
    portfolio_link = "https://anandbheesetti.wixsite.com/portfolio"
    st.markdown(portfolio_link, unsafe_allow_html=True)
    gmail_emoji = "\U0001F4E7"
    st.markdown(f"email me at {gmail_emoji}")
    st.write("anandbheesetti@gmail.com")


st.title("Resume Screening App")
uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])
if uploaded_file is not None:
    
    resume_text=""
    try:
        
        resume_bytes = uploaded_file.read()
        resume_text = resume_bytes.decode('utf-8')
    except UnicodeDecodeError:
        resume_text = resume_bytes.decode('latin-1')
cleaned_resume = clean_resume(resume_text)
input_features = tfidfd.transform([cleaned_resume])
prediction_id = clf.predict(input_features)[0]
st.write(prediction_id)

        # Map category ID to category name
category_mapping = {
  15: "Java Developer",
  23: "Testing",
  8: "DevOps Engineer",
  20: "Python Developer",
  24: "Web Designing",
  12: "HR",
  13: "Hadoop",
  3: "Blockchain",
  10: "ETL Developer",
  18: "Operations Manager",
  6: "Data Science",
  22: "Sales",
  16: "Mechanical Engineer",
  1: "Arts",
  7: "Database",
  11: "Electrical Engineering",
  14: "Health and fitness",
  19: "PMO",
  4: "Business Analyst",
  9: "DotNet Developer",
  2: "Automation Testing",
  17: "Network Security Engineer",
  21: "SAP Developer",
  5: "Civil Engineer",
  0: "Advocate",
}
category_name = category_mapping.get(prediction_id, "Unknown")
st.write("Predicted Category:", category_name)




  
   


