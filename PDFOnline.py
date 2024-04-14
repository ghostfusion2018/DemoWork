import langchain
from langchain.document_loaders import OnlinePDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

# Load the document (replace with your PDF URL)
url = "https://arxiv.org/pdf/2201.08237.pdf"
loader = OnlinePDFLoader(url)
data = loader.load()

# Split the document into chunks for better processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Create a vector store using embeddings from GPT-4All
vectorstore = Chroma.from_documents(documents=all_splits, embedding=GPT4AllEmbeddings())

# Connect to the Ollama server (replace with your Ollama base URL and model name)
ollama = Ollama(base_url="http://localhost:11434", model="llama2")

# Define a prompt template for retrieving and answering questions
prompt_template = langchain.PromptTemplate(
    input_variables=["context", "question"],
    template="Use the following pieces of text to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum and keep the answer as concise as possible. {context} Question: {question} Answer:",
)

# Create a RetrievalQA chain that retrieves relevant parts of the document and uses Ollama to answer the question
qa_chain = RetrievalQA(
    vectorstore=vectorstore,
    llm=ollama,
    prompt_template=prompt_template,
    top_k=3,  # Retrieve top 3 most relevant passages
)

# Now you can ask questions about the document!
question = "What are the motivations behind the research?"
answer = qa_chain.run(context="", question=question)
print(f"Answer: {answer}")
