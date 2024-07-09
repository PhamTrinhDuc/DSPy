from dspy_run import RAGDSPy
import streamlit as st


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