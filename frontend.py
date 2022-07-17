import multiprocessing
import streamlit as st
from app import *


st.title("ARTICLE GENERATOR")
text = st.text_input("Text", value="")
t = multiprocessing.Process(target=Generator, args=(text))
t.start()
