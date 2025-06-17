import sys
from groq import Groq

try:
  from app import get_embedding, index, console, GROQ_API_KEY
except ImportError:
  print("Make sure 'app.py' is in the same folder")
  sys.exit(1)

def main():
  console.rule("Google Drive Documents Chat")
  console.print("Type 'exit' to quit.")

  while True:
    query = console.input("Your Question>")
    if query.lower() in ("exit", "quit"):
      break

    answer = chat_agent(query)
    console.print(f"\nAnswer: {answer}\n")

def chat_agent(query: str) -> str:
  system_message = (
    "You are a helpful assistant that will answer questions based on details received from a google drive document"
    "Retrieve relevant information from the provided internal documents and provide a concise answer."
    "If the answer cannot be found in the provided document, say 'I cannot find the answer in the available resources.'"
  )

  query_embedding = get_embedding(query)
  if query_embedding is None:
    return "Error obtaining query embedding."

  try: 
    result = index.query(
      vector=query_embedding,
      top_k=3,
      include_metadata=True,
      namespace="default"
    )
  except Exception as e:
    return f"Error querying Pinecone: {str(e)}"

  if not result or "matches" not in result or not result["matches"]:
    return "I cannot find the answer in the available resources."

  matches = result["matches"]
  context = " ".join(match["metadata"].get("text", "") for match in matches)
  if not context.strip():
    return "I cannot find the answer in the available resources."

  return groq_chat(system_message, query, context)

  
def groq_chat(system_message: str, query: str, context: str) -> str:
  """
  Calls Groq's chat completion with a system message and users query
  """
  client = Groq(api_key=GROQ_API_KEY)

  try:
    chat_completion = client.chat.completions.create(
      messages=[
        {
          "role": "system",
          "content": f"{system_message}\n\nContext: {context}"
        },
        {
          "role": "user",
          "content": query
        }
      ],
      model="deepseek-r1-distill-llama-70b",
      temperature=0.2,
      max_tokens=512,
    )
    return chat_completion.choices[0].message.content
  except Exception as e:
    return f"Error: {str(e)}"


if __name__ == "__main__":
  main()