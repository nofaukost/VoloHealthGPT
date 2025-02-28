import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import json

# Initialize ChatOllama with default model parameters
@st.cache_resource
def get_chat_model():
    return ChatOllama(model="llama3.1:8b")

chat = get_chat_model()

# Define a dictionary to map conditions to their respective prompt templates
prompts = {
    "anxiety": ChatPromptTemplate.from_messages([
        ("system", """You are a compassionate AI psychologist focused on helping with anxiety and panic attacks. 
        You aim to:
        1. Offer calming techniques and encourage mindfulness to help the user manage their anxiety.
        2. Validate the user's feelings and reassure them that anxiety is a common experience.
        3. Suggest small, actionable steps like deep breathing or grounding exercises to regain control.
        4. Engage in a gentle, supportive conversation that reduces the user's immediate distress."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]),
    "depression": ChatPromptTemplate.from_messages([
        ("system", """You are a supportive AI psychologist assistant focused on helping users navigate through depression. 
        You should:
        1. Provide empathetic support, validating the user's emotions and experiences.
        2. Encourage small, positive actions, even if they seem difficult, to help the user feel a sense of accomplishment.
        3. Remind the user that they are not alone, and that reaching out for support is a sign of strength.
        4. Maintain a warm, understanding tone that fosters hope and connection."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]),
    "stress": ChatPromptTemplate.from_messages([
        ("system", """You are an AI psychologist assistant dedicated to helping users manage stress and avoid burnout. 
        Your role includes:
        1. Offering practical advice on stress management, including relaxation techniques and time management tips.
        2. Acknowledging the user's challenges and validating their feelings of overwhelm.
        3. Encouraging the user to take breaks, set boundaries, and prioritize self-care.
        4. Engaging in a thoughtful, calming conversation that reduces the user's sense of pressure."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]),
    "relationships": ChatPromptTemplate.from_messages([
        ("system", """You are an empathetic AI psychologist assistant focused on helping users with relationship concerns and feelings of loneliness. 
        You should:
        1. Offer thoughtful, supportive advice on improving communication and understanding in relationships.
        2. Validate the user's feelings and experiences, providing reassurance that they are not alone.
        3. Suggest ways to build and maintain meaningful connections with others.
        4. Foster a warm, inclusive conversation that encourages the user to reflect on their relationships."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]),
    "self-esteem": ChatPromptTemplate.from_messages([
        ("system", """You are an encouraging AI psychologist assistant focused on boosting self-esteem and self-worth. 
        Your goals are to:
        1. Help the user recognize and appreciate their strengths and positive qualities.
        2. Suggest exercises to build self-confidence, such as positive affirmations or setting small goals.
        3. Reassure the user that they are valuable and deserving of respect, just as they are.
        4. Engage in an uplifting conversation that inspires self-compassion and self-acceptance."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]),
    "ocd": ChatPromptTemplate.from_messages([
        ("system", """You are an AI psychologist assistant focused on supporting users with Obsessive-Compulsive Disorder (OCD). 
        Your role is to:
        1. Provide non-judgmental support for managing intrusive thoughts and compulsive behaviors.
        2. Suggest practical techniques like exposure and response prevention (ERP).
        3. Encourage the user to seek professional help when necessary.
        4. Maintain a calm, understanding tone that helps the user feel safe and supported."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]),
    "addiction": ChatPromptTemplate.from_messages([
        ("system", """You are an empathetic AI psychologist assistant dedicated to supporting users struggling with addiction and substance use. 
        Your goals are to:
        1. Provide compassionate, non-judgmental support to the user.
        2. Encourage the user to reflect on their triggers and develop healthier coping mechanisms.
        3. Suggest professional resources and support groups as part of the recovery process.
        4. Engage in a gentle, supportive conversation that emphasizes hope and progress."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]),
    "sleep": ChatPromptTemplate.from_messages([
        ("system", """You are a caring AI psychologist assistant focused on helping users with sleep issues. 
        Your role includes:
        1. Offering advice on improving sleep hygiene, such as maintaining a consistent sleep schedule and reducing screen time before bed.
        2. Suggesting relaxation techniques to help the user unwind and prepare for sleep.
        3. Reassuring the user that occasional sleep disturbances are common and manageable.
        4. Engaging in a calming, reassuring conversation that helps the user feel more at ease about their sleep concerns."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]),
    "suicidal": ChatPromptTemplate.from_messages([
        ("system", """You are an AI psychologist assistant focused on providing immediate support to users experiencing suicidal thoughts or self-harm. 
        Your goals are to:
        1. Offer hope and remind the user that help is available and they are not alone.
        2. Encourage the user to reach out to a trusted person or professional.
        3. Provide contact information for emergency services or suicide prevention hotlines if necessary.
        4. Engage in a compassionate, life-affirming conversation that focuses on the user's immediate safety."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]),
    "bipolar": ChatPromptTemplate.from_messages([
        ("system", """You are an AI psychologist assistant focused on helping users manage bipolar disorder and mood swings. 
        Your role includes:
        1. Offering support for managing mood fluctuations, emphasizing stability and self-care.
        2. Encouraging the user to maintain a mood diary to track patterns and triggers.
        3. Suggesting professional help for medication management and therapy.
        4. Engaging in a balanced, understanding conversation that respects the complexity of the user's experience."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]),
    "psychosis": ChatPromptTemplate.from_messages([
        ("system", """You are a compassionate AI psychologist assistant focused on supporting users experiencing psychosis or schizophrenia. 
        Your goals are to:
        1. Provide non-judgmental, empathetic support to the user.
        2. Encourage the user to seek immediate help from a mental health professional.
        3. Reassure the user that they are not alone and that help is available.
        4. Maintain a calm, grounding tone that helps the user feel safe and understood."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]),
    "adhd": ChatPromptTemplate.from_messages([
        ("system", """You are a practical AI psychologist assistant focused on helping users with ADHD and concentration issues. 
        Your role includes:
        1. Offering strategies for improving focus and productivity, such as breaking tasks into smaller steps and using reminders.
        2. Encouraging the user to find methods that work best for them, acknowledging that everyone is different.
        3. Suggesting professional evaluation and support if necessary.
        4. Engaging in a supportive, constructive conversation that empowers the user to manage their symptoms."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]),
    "eating_disorders": ChatPromptTemplate.from_messages([
        ("system", """You are an empathetic AI psychologist assistant focused on supporting users with eating disorders. 
        Your goals are to:
        1. Provide compassionate, non-judgmental support without focusing on weight or appearance.
        2. Encourage the user to seek help from a therapist specializing in eating disorders.
        3. Remind the user that recovery is possible and that they are worthy of care and compassion.
        4. Engage in a gentle, understanding conversation that fosters self-acceptance and healing."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]),
    "general_support": ChatPromptTemplate.from_messages([
        ("system", """You are an empathetic and supportive AI psychologist assistant. 
        Your role includes:
        1. Showing empathy and validating the user's experiences and emotions.
        2. Maintaining a non-judgmental, compassionate approach throughout the conversation.
        3. Focusing on recommending content created by human experts when appropriate.
        4. Serving as a supportive guide, reminding the user that they are not alone.
        5. Maintaining professional boundaries and never flirting or sexualizing the conversation.
        6. If self-harm or suicidal thoughts are mentioned, gently offering hope and guiding the user toward professional resources.
        7. Encouraging self-compassion and resilience in all interactions."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
}

# Function to use LLM to classify the input
def classify_input(user_input):
    classifier_prompt = """You are an AI assistant that classifies user input into one of the following categories:
    1. Anxiety
    2. Depression
    3. Stress
    4. Relationships
    5. Self-Esteem
    6. OCD
    7. Addiction
    8. Sleep Issues
    9. Suicidal Thoughts
    10. Bipolar
    11. Psychosis
    12. ADHD
    13. Eating Disorders
    14. General Support

    Classify the following user input into one of these categories and respond with the category name only: {input}
    """
    complete_prompt = classifier_prompt.format(input=user_input)
    classification_chat = ChatOllama(model="llama3.1:8b", temperature=0)
    classifier_response = classification_chat.invoke(complete_prompt)

    # Map the detailed response to the simplified key used in prompts
    detailed_category = classifier_response.content.strip().lower()

    # Create a mapping from the detailed category to the simplified prompt key
    category_mapping = {
        "anxiety and panic attacks": "anxiety",
        "depression": "depression",
        "stress and burnout": "stress",
        "relationships and loneliness": "relationships",
        "self-esteem and self-worth": "self-esteem",
        "obsessive-compulsive disorder (ocd)": "ocd",
        "addiction and substance use": "addiction",
        "sleep issues": "sleep",
        "suicidal thoughts and self-harm": "suicidal",
        "bipolar disorder and mood swings": "bipolar",
        "psychosis and schizophrenia": "psychosis",
        "adhd and concentration issues": "adhd",
        "eating disorders": "eating_disorders",
        "general support": "general_support"
    }

    # Return the simplified category name
    simplified_category = category_mapping.get(detailed_category, "general_support")

    return json.dumps({"category": simplified_category})

# Function to reset the chat
def reset_chat():
    st.session_state.messages = []
    st.session_state.chat_history = ChatMessageHistory()
    st.session_state.new_chat = True

# Setup RAG
@st.cache_resource
def setup_rag():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma(persist_directory="/home/ubuntu/PsycoGPT/notebooks/mydb", embedding_function=embeddings)

    template = """You are an empathetic and supportive AI psychologist assistant. Use the following pieces of context to answer the human's question. Show empathy and validating the user's experiences and emotions.
    If self-harm or suicidal thoughts are mentioned, gently offering hope and guiding the user toward professional resources.

    Context: {context}

    Human: {input}
    AI Assistant: """

    prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": RunnablePassthrough(), "input": RunnablePassthrough()}
        | prompt
        | chat
        | StrOutputParser()
    )

    return rag_chain, vectorstore

rag_chain, vectorstore = setup_rag()

# Function to reset the chat
def reset_chat():
    st.session_state.messages = []
    st.session_state.chat_history = ChatMessageHistory()
    st.session_state.new_chat = True
    st.session_state.sources = []
    st.session_state.chunks = []
    st.session_state.selected_source = None
    # Clear the sidebar
    st.sidebar.empty()

# Define a relevance threshold
RELEVANCE_THRESHOLD = 0.6  # Adjust as needed

def display_chunk_in_sidebar(source, chunk):
    with st.sidebar:
        st.header(f"Source: {source}")
        st.text_area("Chunk content:", value=chunk, height=300, disabled=True)

def main():
    st.title("AI Psychologist Chatbot")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = ChatMessageHistory()
    if "new_chat" not in st.session_state:
        st.session_state.new_chat = False
    if "sources" not in st.session_state:
        st.session_state.sources = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "selected_source" not in st.session_state:
        st.session_state.selected_source = None

    # New Chat button
    if st.button("New Chat"):
        reset_chat()

    # Chat interface
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        
        # Display source buttons after assistant messages
        if message["role"] == "assistant" and "sources" in message:
            st.write("Sources:")
            cols = st.columns(min(3, len(message["sources"])))
            for i, (source, chunk) in enumerate(zip(message["sources"][:3], message["chunks"][:3])):
                with cols[i]:
                    if st.button(f"Source {i+1}", key=f"source_{msg_idx}_{i}"):
                        st.session_state.selected_source = (msg_idx, i)

    # Chat input
    if prompt := st.chat_input("What's on your mind?"):
        st.session_state.new_chat = False
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Use LLM to classify the user input
        classification_result = classify_input(prompt)
        classification_json = json.loads(classification_result)
        category = classification_json['category']

        # Select the prompt based on the classification
        selected_prompt = prompts.get(category, prompts["general_support"])

        # Retrieve relevant documents with scores
        docs_and_scores = vectorstore.similarity_search_with_score(prompt, k=3)

        # Filter documents based on the relevance threshold
        relevant_docs = [doc for doc, score in docs_and_scores if score >= RELEVANCE_THRESHOLD]

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # If relevant documents are found, use RAG
            if relevant_docs:
                context = "\n".join([doc.page_content for doc in relevant_docs])
                for chunk in rag_chain.stream({"context": context, "input": prompt}):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")

                sources = [doc.metadata.get('source', 'Unknown') for doc in relevant_docs]
                chunks = [doc.page_content for doc in relevant_docs]
            else:
                # Use the existing chain if no relevant documents are found
                chain = selected_prompt | chat
                chain_with_message_history = RunnableWithMessageHistory(
                    chain,
                    lambda session_id: st.session_state.chat_history,
                    input_messages_key="input",
                    history_messages_key="chat_history"
                )
                for chunk in chain_with_message_history.stream(
                    {"input": prompt},
                    {"configurable": {"session_id": "user_session"}}
                ):
                    full_response += chunk.content
                    message_placeholder.markdown(full_response + "▌")

                sources = []
                chunks = []

            message_placeholder.markdown(full_response)

        # Add the new message with sources and chunks
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": sources,
            "chunks": chunks
        })

        # Display source buttons after the new assistant message
        if sources:
            st.write("Sources:")
            cols = st.columns(min(3, len(sources)))
            for i, (source, chunk) in enumerate(zip(sources[:3], chunks[:3])):
                with cols[i]:
                    if st.button(f"Source {i+1}", key=f"source_{len(st.session_state.messages)-1}_{i}"):
                        st.session_state.selected_source = (len(st.session_state.messages)-1, i)

        # Update chat history
        st.session_state.chat_history.add_message(HumanMessage(content=prompt))
        st.session_state.chat_history.add_message(AIMessage(content=full_response))
        st.session_state.chat_history.messages = st.session_state.chat_history.messages[-10:]

    # Display selected source in sidebar
    if st.session_state.selected_source is not None:
        msg_idx, source_idx = st.session_state.selected_source
        if msg_idx < len(st.session_state.messages):
            message = st.session_state.messages[msg_idx]
            if "sources" in message and source_idx < len(message["sources"]):
                display_chunk_in_sidebar(message["sources"][source_idx], message["chunks"][source_idx])

    # If it's a new chat, display a welcome message
    if st.session_state.new_chat:
        st.chat_message("assistant").markdown("Hello! I'm here to listen and support you. How are you feeling today?")

if __name__ == "__main__":
    main()