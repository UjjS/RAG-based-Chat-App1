# Chat with any Website - RAG Application

This is a full-fledged RAG (Retrieval-Augmented Generation) based chat application that allows you to have a conversation with the content of any public website.

You provide a URL, the application scrapes its content, indexes it into a vector database, and lets you ask questions about it using a powerful language model.

## Features

- **Conversational AI**: Chat with a document, ask follow-up questions.
- **Web-based UI**: Clean and simple interface powered by Streamlit.
- **Efficient**: Caches processed data to avoid re-computing for the same URL.
- **Containerized**: Easy to set up and run using Docker and Docker Compose.
- **Persistent Storage**: Vector data is saved to a local volume, so it persists across container restarts.

## Project Structure