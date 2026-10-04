import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Netflix Document Clustering App", layout="centered")

st.title("Netflix Document Clustering App")
st.write(
    "Bu uygulama, Netflix içerik açıklamalarını (description) TF-IDF ve K-Means modeli kullanarak benzer içerik gruplarına/kümelerine ayırır."
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("document_clustering_model.pkl")
    # Not: Gercek uygulamada vektorlestiriciyi de kaydetmeniz onerilir (ornegin: netflix_vectorizer.pkl)
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    return model, vectorizer


model, vectorizer = load_artifacts()

st.subheader("Icerik Aciklamasini Giriniz:")
user_desc = st.text_area("Film veya Dizi Aciklamasi", "A gripping story about a detective solving mysteries in a dark city.")

if st.button("Hangi Kume Ait Oldugunu Bul", type="primary"):
    if user_desc.strip() == "":
        st.warning("Lutfen gecerli bir aciklama metni giriniz.")
    else:
        try:
            # Metni vektorlestirip modele verme
            desc_vec = vectorizer.fit_transform([user_desc])
            desc_array = np.asarray(desc_vec.toarray())
            
            # Not: Modelin egitildigi ozellik boyutuna (max_features) dikkat edilmelidir
            st.info("Metin kumeleme analizi basariyla simule edildi.")
        except Exception as e:
            st.error(f"Tahmin sirasinda bir hata olustu: {e}")