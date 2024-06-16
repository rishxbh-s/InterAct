import streamlit as st
from llama_index.legacy import (
    SimpleDirectoryReader, VectorStoreIndex, ServiceContext
)
from llama_cpp import Llama
from llama_index.legacy.llms.llama_utils import (
    messages_to_prompt, completion_to_prompt
)
from langchain.schema import SystemMessage, HumanMessage, AIMessage

def init_page() -> None:
    st.set_page_config(page_title="Personal Chatbot")
    st.header("ChatGuru")
    st.sidebar.title("Options")

def select_llm() -> Llama:
    return Llama(
        model_path="/content/llama-2-7b-chat.Q2_K.gguf",
        temperature=0.1,
        max_new_tokens=500,
        context_window=3900,
        generate_kwargs={},
        model_kwargs={"n_gpu_layers": 1},
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )

def init_messages() -> None:
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful AI assistant.")
        ]

def get_persona() -> str:
    persona = st.sidebar.selectbox(
        "Select a persona",
        (
            "General",
            "Kitchen Helper",
            "Travel Planner",
            "Personal Trainer",
            "Tech Support",
            "Creative Writer",
        ),
    )
    return persona

def customize_persona(persona: str) -> None:
    if persona == "Kitchen Helper":
        st.session_state.messages[0].content = (
            "You are a friendly and knowledgeable kitchen helper, "
            "providing cooking tips, recipes, and advice on kitchen tools and appliances."
        )
    elif persona == "Travel Planner":
        st.session_state.messages[0].content = (
            "You are an experienced travel planner, offering recommendations "
            "for destinations, itineraries, accommodations, and cultural experiences."
        )
    elif persona == "Personal Trainer":
        st.session_state.messages[0].content = (
            "You are a motivational personal trainer, providing workout plans, "
            "nutrition advice, and encouraging a healthy lifestyle."
        )
    elif persona == "Tech Support":
        st.session_state.messages[0].content = (
            "You are a knowledgeable tech support specialist, troubleshooting "
            "issues, offering software and hardware recommendations, and providing guidance on tech-related topics."
        )
    elif persona == "Creative Writer":
        st.session_state.messages[0].content = (
            "You are an imaginative creative writer, assisting with storytelling, "
            "character development, writing prompts, and providing feedback on creative works."
        )
    else:
        st.session_state.messages[0].content = "You are a helpful AI assistant."

def get_answer(llm, messages) -> str:
    response = llm.complete(messages)
    return response

def main() -> None:
    init_page()
    llm = select_llm()
    init_messages()
    persona = get_persona()
    customize_persona(persona)

    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("Bot is typing ..."):
            answer = get_answer(llm, st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=answer))

    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)

if __name__ == "__main__":
    main()
