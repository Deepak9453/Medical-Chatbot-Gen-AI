from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

os.environ["PINECONE_API_KEY"] = "pcsk_34h5tB_7w5M2gFgLaJ4biBGF9oAq28MvFsaqRf4Bt3pFE2UreFUrbMNHmtAtBqCER**********"

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')

#PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
#GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

PINECONE_API_KEY="pcsk_34h5tB_7w5M2gFgLaJ4biBGF9oAq28MvFsaqRf4Bt3pFE2UreFUrbMNHmtAtBq*************"
GOOGLE_API_KEY='AIzaSyDm4DYOo-1ohGNnOEHUlbuJ0***********'

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})


#llm = OpenAI(temperature=0.4, max_tokens=500)

# Set the model (assuming you're using gpt-3.5-turbo or a similar Google model)
MODEL = "gemini-1.5-flash" 

# Set up the Google Generative AI model with desired parameters
llm = ChatGoogleGenerativeAI(
    model=MODEL,
    temperature=0.4,
    max_tokens=500
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])




if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
