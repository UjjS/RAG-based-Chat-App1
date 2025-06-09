import os
import streamlit as st
from dotenv import load_dotenv
from loader import get_text_from_url, get_text_chunks
from rag_pipeline import get_vector_store, get_conversation_chain

def handle_user_input(user_question):
    """Processes user input, gets a response from the chain, and updates the session state."""
    if st.session_state.conversation:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

            # Display chat history
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.chat_message("user").write(message.content)
                else:
                    st.chat_message("assistant").write(message.content)
    else:
        st.warning("Please process a URL first.")

def main():
    # Load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OPENAI_API_KEY not found. Please set it in your .env file.")
        return

    # Page configuration
    st.set_page_config(page_title="Chat with any Website", page_icon="🌐")

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processed_url" not in st.session_state:
        st.session_state.processed_url = ""

    # --- UI LAYOUT ---
    st.header("Chat with any Website 🌐")

    # Display chat messages from history
    if st.session_state.chat_history:
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.chat_message("user").write(message.content)
            else:
                st.chat_message("assistant").write(message.content)

    # Sidebar for URL input and processing
    with st.sidebar:
        st.subheader("Your Document")
        url_input = st.text_input("Enter the URL of the website you want to chat with:")

        if st.button("Process URL"):
            if url_input:
                if url_input != st.session_state.processed_url:
                    with st.spinner("Processing URL... This may take a moment."):
                        # 1. Get text from URL
                        raw_text = get_text_from_url(url_input)
                        if not raw_text:
                            st.error("Failed to retrieve content from the URL. Please check the URL and try again.")
                            return

                        # 2. Get text chunks
                        text_chunks = get_text_chunks(raw_text)
                        
                        # 3. Create vector store (cached)
                        vector_store = get_vector_store(text_chunks, openai_api_key)
                        
                        # 4. Create conversation chain
                        st.session_state.conversation = get_conversation_chain(vector_store, openai_api_key)
                        st.session_state.processed_url = url_input
                        st.session_state.chat_history = None # Reset chat history
                        
                    st.success("Processing Complete! You can now ask questions.")
                else:
                    st.info("This URL has already been processed.")
            else:
                st.warning("Please enter a URL.")

    # Chat input at the bottom
    if user_question := st.chat_input("Ask a question about the website's content..."):
        handle_user_input(user_question)

if __name__ == '__main__':
    main()