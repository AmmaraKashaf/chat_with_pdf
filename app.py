import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

# Step 1: Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Step 2: Load PDF
loader = PyMuPDFLoader("your_pdf.pdf")  # 📝 change if file name is different
documents = loader.load()

# Step 3: Split text
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

# Step 4: Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Step 5: Setup Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

# Step 6: Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

# Step 7: Start Chat
print("Ask a question about the PDF (or type 'exit'):")

while True:
    query = input("> ")
    if query.lower() == "exit":
        break
    answer = qa_chain.run(query)
    print("Answer:", answer)

