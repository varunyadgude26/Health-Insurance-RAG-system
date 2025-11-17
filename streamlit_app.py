import streamlit as st
import requests
import json
import os


FASTAPI_URL = "https://health-insurance-rag-system-vus6.onrender.com"  

st.set_page_config(page_title="Health Insurance RAG", layout="wide")

st.title("Health Insurance Policy Enquiry System (Gemini + Pinecone + RAG)")
st.write("Upload your insurance documents and ask policy-related questions.")



st.header("Upload Policy Documents")

uploaded_files = st.file_uploader(
    "Upload PDF / DOCX / TXT / MD files",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)

if st.button("Upload to System"):
    if not uploaded_files:
        st.warning("Please upload one or more files first.")
    else:
        with st.spinner("Uploading to FastAPI and indexing into Pinecone..."):
            files = [("files", (f.name, f.read(), f"type")) for f in uploaded_files]

            try:
                response = requests.post(
                    f"{FASTAPI_URL}/system-upload-documents",
                    files=files
                )
                if response.status_code == 200:
                    st.success("Documents uploaded and indexed successfully!")
                    st.json(response.json())
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")


st.header("Ask a Question About Your Policy")

question = st.text_input("Enter your insurance-related question:")

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please type a question.")
    else:
        with st.spinner("Querying RAG system..."):
            try:
                response = requests.post(
                    f"{FASTAPI_URL}/ask-question",
                    json={"question": question}
                )
                result = response.json()

                st.subheader("Answer")
                st.write(result.get("answer", "No answer returned."))

                st.subheader("Sources Used")
                sources = result.get("sources", [])
                if sources:
                    for src in sources:
                        st.markdown(f"""
                        **File:** {src.get('file_name')}  
                        **Page:** {src.get('page_no')}  
                        **Chunk_id:** {src.get('chunk_index')}
                        ---
                        """)
                else:
                    st.info("No citations returned.")

                st.subheader("Confidence Score:")
                st.write(result.get("confidence"))

            except Exception as e:
                st.error(f"Error: {str(e)}")


st.write("---")
st.caption("Built with using Streamlit + FastAPI + Gemini + Pinecone")
