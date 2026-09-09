from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.graph.message import add_messages
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langgraph.checkpoint.sqlite import SqliteSaver 
from dotenv import load_dotenv
import sqlite3



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


conn = sqlite3.connect(database='chatbot.db', check_same_thread= False)

checkpointer = SqliteSaver(conn = conn)
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer = checkpointer)


# test
# response  = chatbot.invoke( 
#                 {'messages' : [HumanMessage(content= "What i asked last ?")]},
#                 config = {'configurable': {'thread_id':"thread-1"}},
#              )

# print(response)

# test
# response  = chatbot.invoke( 
#                 {'messages' : [HumanMessage(content= "Hi my name is sumitt ?")]},
#                 config = {'configurable': {'thread_id':"thread-1"}},
#              )

# print(chatbot.get_state(config={'configurable':{'thread_id':'thread-1'}}).values['messages'])

#extracting all unique threads stored in db

def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return (list(all_threads))