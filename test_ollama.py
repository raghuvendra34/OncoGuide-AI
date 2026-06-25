from ollama import chat

response = chat(
    model="llama3",
    messages=[
        {
            "role": "user",
            "content": "Say hello"
        }
    ]
)

print(response["message"]["content"])