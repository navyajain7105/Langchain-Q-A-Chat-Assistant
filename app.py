import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal, Optional

st.title("🧠 LangChain Q&A Assistant")
# To load api keys from .env file
load_dotenv()

# Initialize chat memory only once
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

# Model type selection
model_type = st.selectbox("Choose Model Type:", ["Open-Source", "Closed-Source (Groq)"])

# Model name selection
if model_type == "Open-Source":
    model_name = st.selectbox("Choose Open-Source Model:", ["HuggingFace - API", "HuggingFace - Local"])
else:
    model_name = st.selectbox("Choose Groq Model:", ["Groq-LLaMA2", "Groq-Mistral"])

# Chat_history
chat_history = st.selectbox("Do you want to save chat history:", ["Yes", "No"])

# User input
user_input = st.text_input("Ask a question:")

Submit=st.button('Submit')
Reset=st.button("Reset Chat")

if Reset:
    st.session_state.chat_memory = []
    st.success("Chat history cleared!")
ChatHist = st.checkbox("Chat History")

if ChatHist:
    # Loops through the list of messages we saved in session state.
    for msg in st.session_state.chat_memory:
        if isinstance(msg, HumanMessage):
            # st.chat_message("user👨🏻‍💻").write(msg.content)
            st.write("user👨🏻‍💻 : ",msg.content)
            # Displays the message in the chat UI as if a user sent it.
        elif isinstance(msg, AIMessage):
            # st.chat_message("assistant🤖").write(msg.content)
            st.write("assistant🤖 : ",msg.content)

# if agree:
#     st.write("Great!")

# Output Parser Schema
class Schema(BaseModel):
        Answer: str = Field(description="Answer of the user input question")
        Description: str = Field(description="Explaination of the user input question")
        # Model_type: Optional[Literal["Open-Source","Closed-Source (Groq)"]] = Field(default="none",description="Type of the model used")
        Model_type: Optional[str] = Field(default="none",description="Type of the model used")
        # Model_name: Optional[Literal["HuggingFace-API","HuggingFace-Local","Groq-LLaMA2","Groq-Mistral"]] = Field(default="none",description="Name of the model used")
        Model_name: Optional[str] = Field(default="none",description="Name of the model used")

# pydantic object parser fron groq
parser = PydanticOutputParser(pydantic_object=Schema)
parser1 = StrOutputParser()

if user_input:
    # === BACKEND LOGIC STARTS HERE ===

    # 1. Load selected model
    # TODO: Write logic to load the selected model dynamically based on dropdown
    def get_model():
        if model_type == "Open-Source" and model_name == "HuggingFace - API":
            llm = HuggingFaceEndpoint(
                repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                task="text-generation",
            )
            # LLM config

            model = ChatHuggingFace(llm=llm)
            return model

        elif model_type == "Open-Source" and model_name == "HuggingFace - Local":
            llm = HuggingFacePipeline.from_model_id(
                model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                task= "text-generation",
                pipeline_kwargs=dict(
                    max_new_tokens = 500,
                    temperature = 0.5,
                    do_sample = True
                )

            )
            model = ChatHuggingFace(llm=llm)
            return model
        
        elif model_type == "Closed-Source (Groq)" and model_name == "Groq-LLaMA2":
            model = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.0,
                max_tokens= 100
                # other params...
            )
            return model

        else:
            model = ChatGroq(
                model="mistral-saba-24b",
                temperature=0.0,
                max_tokens= 100
                # other params...
            )
            return model

    model = get_model()
    # 2. Create prompt using PromptTemplate or ChatPromptTemplate
    # TODO: Write logic to construct the appropriate prompt

    template = PromptTemplate(
        # template='You are a helpful assistant using a {model_type} model named {model_name}\n.Answer the following question {user_input}\n {format_instruction}',
        # template='You are a helpful assistant. Answer the following question {user_input},Model_type: {model_type}Model_name: {model_name}\n',
        template="""You are a helpful assistant using a {model_type} model named {model_name}. 
Answer the following question in a valid JSON format that strictly adheres to the schema below. 
Do not include any extra text, placeholders, or explanations outside the JSON object.

Question: {user_input}

Schema:
{format_instruction}

Example response:
{{"Answer": "Example answer", "Description": "Example description", "Model_type": "{model_type}", "Model_name": "{model_name}"}}""",
        input_variables=["user_input", "model_type", "model_name"],
        # validate_template=True,
        partial_variables={'format_instruction':parser.get_format_instructions()}
    )

    template1 = PromptTemplate(
        template='You are a helpful assistant. Answer the following question {user_input}',
        input_variables=["user_input"]
    )

    # 3. Maintain previous chat history (optional: use MessagesPlaceholder equivalent)
    # TODO: Implement chat memory logic

    format_instructions = parser.get_format_instructions()

    chat_template = ChatPromptTemplate.from_messages([
        # SystemMessage(content="You are a helpful Q&A Assistant."),
        # SystemMessage(content="You are a helpful assistant using a model type {model_type} and model named {model_name}. Please format your response as follows:\n{format_instructions}"),
        SystemMessage(content="""You are a helpful assistant using a {model_type} model named {model_name}. 
Respond to the user's question in a valid JSON format that strictly adheres to the schema below. 
Do not include any extra text or explanations outside the JSON object.

Schema:
{format_instructions}

Example response:
{{"Answer": "Example answer", "Description": "Example description", "Model_type": "{model_type}", "Model_name": "{model_name}"}}"""),
        MessagesPlaceholder(variable_name='chat_history'),
        HumanMessage(content="{question}")
    ]).partial(format_instructions=format_instructions)

    chat_template1 = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful Q&A Assistant."),
        MessagesPlaceholder(variable_name='chat_history'),
        HumanMessage(content="{question}")
    ])
    
    # 4. Parse the output
    # TODO: Use output parser to clean and display response

    if chat_history == "No" :
        if model_type == "Open-Source":
            chain = template1 | model | parser1
        else:
            chain = template | model | parser
    else:
        if model_type == "Open-Source":
            chain = chat_template1 | model | parser1
        else:
            chain = chat_template | model | parser

    

# 5. Display the response
# TODO: Show the final answer to the user

if Submit:
    if user_input:
        # Check if chat history is set to "No" and process accordingly
        if chat_history == "No" :
            if model_type == "Closed-Source (Groq)":
                response = chain.invoke({'user_input': user_input, 'model_type': model_type, 'model_name': model_name})
                st.write("Your Result: \n\n", response)
                st.session_state.chat_memory = []  # Clear chat memory after processing
            elif model_type == "Open-Source":
                response = chain.invoke({'user_input': user_input})
                st.write("Your Result: \n\n", response)
                st.session_state.chat_memory = []  # Clear chat memory after processing
        else:
            # Optional: handle the case where chat history is "Yes"
            if model_type == "Closed-Source (Groq)":
                st.session_state.chat_memory.append(HumanMessage(content=user_input))
                response = chain.invoke({'user_input': user_input,"chat_history": st.session_state.chat_memory,'model_type': model_type,'model_name': model_name})
                st.session_state.chat_memory.append(AIMessage(content='Answer: '+response.Answer+'\nDescription: '+response.Description))
                st.write("Your Result: \n\n",response)

            elif model_type == "Open-Source":
                st.session_state.chat_memory.append(HumanMessage(content=user_input))
                response = chain.invoke({'user_input': user_input,"chat_history": st.session_state.chat_memory})
                st.session_state.chat_memory.append(AIMessage(content=response))
                st.write("Your Result: \n\n",response)
    else:
        st.warning("Enter Your Question")  # Handle case where input is empty

