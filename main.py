import streamlit as st
import file
import sidebar
import text
import tweet
import tweeter
import Reddit


# st.title("Hello")
page = sidebar.show()

if page=="File":
    file.renderPage()
elif page=="Text":
    text.renderPage()
elif page=="Tweet":
    Reddit.file_page()

# elif page=="IMDb movie reviews":
#     imdbReviewsPage.renderPage()
# elif page=="Image":
#     imagePage.renderPage()
