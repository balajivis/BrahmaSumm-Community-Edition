# from langchain_text_splitters import TokenTextSplitter

# text_splitter = TokenTextSplitter(chunk_size=10, chunk_overlap=0)

# texts = text_splitter.split_text("heloo welcone otd jvbk =works")
# print(texts)



























from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOllama(
    model="gemma2:2b",
    temperature=0,
)



print(llm.invoke([
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]))