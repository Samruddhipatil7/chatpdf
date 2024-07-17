import streamlit as st
import pdfplumber
from transformers import pipeline, RagTokenizer, RagRetriever, RagSequenceForGeneration

def preprocess_text(text):
    # Remove extra whitespace and normalize line breaks
    text = text.replace('\n', ' ').replace('\r', '')
    text = ' '.join(text.split())
    return text

st.title("Chat with Your PDF")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner('Reading PDF...'):
        # Extract text from PDF using pdfplumber
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

        text = preprocess_text(text)
        st.success('PDF successfully read and preprocessed!')

        # Display some text from the PDF
        st.text_area("Extracted Text", text[:1000], height=300)

        # Initialize the RAG model
        tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
        retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)
        rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")

        # Tokenize the text for RAG
        input_texts = text.split('. ')
        input_ids = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Build context embeddings for retrieval
        context_input_ids = retriever(input_ids.input_ids, input_ids.input_ids, num_beams=2)

        question = st.text_input("Ask a question about the PDF:")
        if question:
            with st.spinner('Searching for answer...'):
                # Tokenize the question
                question_ids = tokenizer(question, return_tensors="pt")['input_ids']
                
                # Generate answer using RAG
                generated = rag_model.generate(input_ids=context_input_ids.input_ids, context_input_ids=question_ids, num_beams=2)
                rag_answer = tokenizer.decode(generated[0], skip_special_tokens=True)
                st.write(rag_answer)
