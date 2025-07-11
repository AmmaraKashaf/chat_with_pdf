import os
import fitz  
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai


os.environ["GOOGLE_API_KEY"] = "AIzaSyAsr0XzqKoXXnmzl7q7z2774TewSdFOvms"  # Use dotenv in real projects!
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


doc = fitz.open("paper.pdf")
text = ""
for page in doc:
    text += page.get_text()


splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
texts = splitter.split_text(text)


embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_texts(texts, embedding_model, persist_directory="./chroma_db")
retriever = vectorstore.as_retriever()

print(" PDF chunks embedded and saved in vector DB.")


model = genai.GenerativeModel("gemini-2.0-flash") 


while True:
    query = input("\nAsk a question about the PDF (or type 'exit'): ")
    if query.lower() == "exit":
        break

    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a helpful assistant. Use the content from the PDF to answer the question.

PDF Content:
{context}

Question: {query}
Answer:
"""

    try:
        response = model.generate_content(prompt)
        print(" Answer:", response.text)
    except Exception as e:
        print(" Error talking to Gemini:", str(e))


