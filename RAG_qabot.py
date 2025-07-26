from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

import os
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

# Disable Chroma telemetry
os.environ["CHROMA_TELEMETRY"] = "False"

# --------- 1. LLM ----------
def get_llm():
    model_id = 'ibm/granite-3-3-8b-instruct'
    parameters = {
        GenParams.MAX_NEW_TOKENS: 512,
        GenParams.TEMPERATURE: 0.5,
    }
    return WatsonxLLM(
        model_id=model_id,
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=parameters,
    )

# --------- 2. Embedding Model ----------
def watsonx_embedding():
    embed_params = {
        EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
        EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True}
    }
    return WatsonxEmbeddings(
        model_id='ibm/slate-125m-english-rtrvr',  # Confirm this exists in Watsonx
        url="https://us-south.ml.cloud.ibm.com",
        project_id="skills-network",
        params=embed_params
    )

# --------- 3. Document Loading ----------
def document_loader(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from PDF.")
    return docs

# --------- 4. Text Splitting ----------
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=8,
        length_function=len
    )
    chunks = splitter.split_documents(data)
    print(f"Split into {len(chunks)} text chunks.")
    print(f'Chunks: {chunks}')
    return chunks

# --------- 5. Vector Store with Embedding Filtering ----------
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    texts = [doc.page_content for doc in chunks]

    embeddings = embedding_model.embed_documents(texts)
    print(f"Embedding output: {len(embeddings)}")

    # Filter out None or invalid embeddings
    valid_chunks = []
    for i, emb in enumerate(embeddings):
        if emb is not None and isinstance(emb, list) and len(emb) > 0:
            valid_chunks.append(chunks[i])
        else:
            print(f"⚠️ Skipping chunk {i} due to invalid embedding.")

    if not valid_chunks:
        raise ValueError("❌ All chunks failed embedding. No valid documents to store.")

    vectordb = Chroma.from_documents(
        documents=valid_chunks,
        embedding=embedding_model,
        persist_directory="./chroma_db"
    )
    vectordb.persist()
    return vectordb

# --------- 6. Retriever ----------
def get_retriever(file_path):
    docs = document_loader(file_path)
    if not docs:
        raise ValueError("❌ The uploaded PDF contains no readable text.")

    chunks = text_splitter(docs)
    if not chunks:
        raise ValueError("❌ Text splitting produced no chunks.")

    vectordb = vector_database(chunks)
    return vectordb.as_retriever()

# --------- 7. QA Chain ----------
def retriever_qa(file_path, query):
    try:
        if not file_path or not query.strip():
            return "⚠️ Please upload a valid PDF and type a question."

        llm = get_llm()
        retriever_obj = get_retriever(file_path)

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type='stuff',
            retriever=retriever_obj,
            return_source_documents=True
        )

        response = qa_chain.invoke(query)
        return response.get('result', "⚠️ No answer returned.")

    except Exception as e:
        return f"❌ Error: {str(e)}"

# --------- 8. Gradio Interface ----------
rag_application = gr.Interface(
    fn=retriever_qa,
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=[".pdf"], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG QA Bot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document.",
    allow_flagging=False
)

# --------- 9. Launch ----------
if __name__ == "__main__":
    rag_application.launch(server_name='127.0.0.1', share=True)
