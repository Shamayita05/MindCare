# main.py
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Mental health model
model = OllamaLLM(model="gemma3:1b")

template = """
You are a compassionate mental health therapist.

Here are some relevant Q&A excerpts from previous conversations:
{reviews}

Here is the user's question to answer:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n-------------------------------")
    question = input("Ask your question (q to quit): ").strip()
    if question.lower() == "q":
        break

    # Retrieve context from mental health Q&A
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print("\nAnswer:\n", result)
