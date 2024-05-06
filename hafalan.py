import streamlit as st
import pandas as pd

def main():
    st.title("Sentiment Analysis")
    tweet = st.text_input("Masukkan Komentar")
    if st.button("Analyze"):
        st.write("anjay")
    
    st.title("Via Files")
    upl = st.file_uploader("File CSV", type=['csv'])
    column = st.text_input("Masukkan Nama Kolom")
    if st.button("Analisi"):
        if upl is not None:
            file = pd.read_csv(upl)
            predictions = []
            predictions.append()


if __name__ == "__main__":
    main()