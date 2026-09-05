from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.memory import MemorySaver 
from dotenv import load_dotenv



load_dotenv()

llm= HuggingFaceEndpoint(
   repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation'
)
model = ChatHuggingFace(llm=llm)


class ChatState(TypedDict):

    messages : Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state['messages']

    response = model.invoke(messages)

    return {'messages': [response]}



checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer = checkpointer)