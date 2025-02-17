import streamlit as st
# Set page configuration including title and icon
st.set_page_config(page_title="ContextLLM",
                   page_icon="🙏")

# Displaying a markdown header with a welcoming message and description
st.markdown("""
            # 🧠 Welcome to ContextLLM! 📑
            Explore the latest insights from the AnnualReport and some pdfs.
            """)

# Displaying a markdown section with instructions on what users can do
st.markdown("""
            ### 🤔 What You Can Do:
            - Go to `CENTIC_RAG` page to ask questions related to the annualreport.pdf
            - 🛠 Working...
            """)