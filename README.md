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

---

You're now ready to run and develop on the app!

