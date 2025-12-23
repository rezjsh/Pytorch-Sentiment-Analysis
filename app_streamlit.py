import streamlit as st
from app.ui import run_app


def main():
    run_app()


if __name__ == "__main__":
    st.set_page_config(page_title="Sentiment Analysis",
                        page_icon="💬",
                        layout="wide",
                        initial_sidebar_state="expanded")
    main()
