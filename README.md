# Setup Instructions

## About the App

This is a **Retrieval-Augmented Generation (RAG)** application. It connects to **Google Drive** to read and retrieve document data, uses **Pinecone** to vectorize the content, and employs the **Deepseek LLM** served via **Groq** to interpret user queries and generate context-aware responses.

## Requirements

- Python 3.x
- `pip` (included with Python 3)

## Installation Guide

### 1. Install Python

Ensure you have **Python 3.x** installed. Download it from the official website: [https://www.python.org/downloads/](https://www.python.org/downloads/)

Verify installation:

```bash
python3 --version
```

### 2. Create Virtual Environment

Run the following command to create a virtual environment:

```bash
python3 -m venv venv
```

### 3. Activate Virtual Environment

On macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

Install all required Python packages using:

```bash
pip install -r requirements.txt
```

### 5. Run the App

After setup is complete, you can start the app with:

```bash
python3 chat_interface.py
```

### 6. Run the App with Live Index Updates

To ensure that your app both serves chat responses and actively updates the Pinecone index:

1. **Terminal 1** – Run the background index updater:

   ```bash
   python3 app.py
   ```

   This process watches for document changes and updates the Pinecone vector store accordingly.

2. **Terminal 2** – Run the chat interface using Deepseek via Groq:

   ```bash
   python3 chat_interface.py
   ```

This dual-terminal setup allows for real-time indexing and responsive chat functionality.

---

You're now ready to run and develop on the app!

