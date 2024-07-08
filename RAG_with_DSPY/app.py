from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import streamlit as st
from dspy.retrieve.qdrant_rm import QdrantRM
import dspy
import os


EMBEDDING_DIM = 384
PATH_DATA = "data/wordWarData.pdf"
MODEL_LLM_NAME = "mixtral-8x7b-32768"
MODEL_EMBEDDING_NAME = "BAAI/bge-small-en-v1.5"

embedding_model = FastEmbedEmbeddings(model_name=MODEL_EMBEDDING_NAME)
client = QdrantClient(":memory:")


def load_documents() -> tuple[list[str], list[int]]:

    """
    Tải dữ liệu PDF, chia nhỏ thành các tài liệu và trả về một list documents và một list ID documents.

    Returns:
        tuple: (list[str], list[int])
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False
    )

    document = PyMuPDFLoader(PATH_DATA)
    loaded_doc = document.load_and_split(text_splitter=text_splitter)
    doc_contents = [doc.page_content for doc in loaded_doc]
    doc_ids = list(range(0, len(loaded_doc)))

    return doc_contents, doc_ids

def embedding_and_store_documents(doc_contents: list[str]) -> None:
    """
    Embedding document được load ở phía trên.
    create collection cho Qdrant và upload document được embed ở phía trên vào.

    Returns:
        None
    """
    # embedding document
    vector_embed = embedding_model.embed_documents(doc_contents)

    # store vector embed to Qdrant

    client.delete_collection(collection_name="cf_data")
    client.create_collection(
        collection_name="cf_data",
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
    )

    client.upload_collection(
        collection_name="cf_data",
        ids=doc_ids,
        vectors=vector_embed
    )

doc_contents, doc_ids = load_documents()
embedding_and_store_documents(doc_contents=doc_contents)

# init retriever
qdrant_retriever_model = QdrantRM(
    qdrant_collection_name="cf_data",
    qdrant_client=client, 
    k=3
)
# init language model from dspy, LM used is called api  Groq
lm = dspy.GROQ(model=MODEL_LLM_NAME, 
               api_key =os.getenv('GROQ_API_KEY'))
dspy.settings.configure(
    rm=qdrant_retriever_model,
    lm=lm
)

# build RAG using components of dspy
class GenerateAnswer(dspy.Signature):
    """Answer questions with logical factoid answers."""

    context = dspy.InputField(desc = "will contain an AI act related document")
    question = dspy.InputField()
    answer = dspy.OutputField(desc = "a answer within 20 to 30 words")

# get context from user's question
def get_context(question: str) -> str:
    query_vector = embedding_model.embed_documents(question)


    hits = client.search(
        collection_name="cf_data",
        query_vector=query_vector[0],
        limit=3
    )

    s=''
    for x in [doc_contents[hit.id] for hit in hits]:
        s = s + x
    return s


class RAGDSPy(dspy.Module):
    def __init__(self, num_passages: int = 3):
        super().__init__()

        self.retriever = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)


    def forward(self, question: str):
        context = get_context(question)
        prediction = self.generate_answer(
            context=context,
            question=question
        )
        return dspy.Prediction(
            context=context,
            answer=prediction.answer
        )

rag = RAGDSPy()
def respond(query: str) -> str:
    response = rag(query)
    return response.answer


# deploy model using streamlit
st.set_page_config(page_title="DSPy RAG Chatbot", page_icon=":robot_face:")


st.markdown("""
<div style="text-align: center;">
            <img src="https://dspy-docs.vercel.app/img/logo.png" alt="Chatbot Logo" width="100"/>
    <img src="https://img.freepik.com/premium-vector/robot-icon-chat-bot-sign-support-service-concept-chatbot-character-flat-style_41737-796.jpg?" alt="Chatbot Logo" width="200"/>
    <h1 style="color: #0078D7;">DSPy based RAG Chatbot</h1>
</div>
""", unsafe_allow_html=True)



st.markdown("""
<p style="text-align: center; font-size: 18px; color: #555;">
    Hello! Just ask me anything from the dataset.
</p>
""", unsafe_allow_html=True)


st.markdown("<hr/>", unsafe_allow_html=True)

user_query = st.text_input("Enter your question:", placeholder="E.g., What is the aim of AI act?")

if st.button("Answer"):
    bot_response = respond(user_query)
   
    st.markdown(f"""
    <div style="background-color: #f9f9f9; padding: 10px; border-radius: 5px; margin-top: 20px;">
        <h4 style="color: #0078D7;">Bot's Response:</h4>
        <p style="color: #333;">{bot_response}</p>
    </div>
    """, unsafe_allow_html=True)