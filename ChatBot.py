import streamlit as st
import time
import openai
import os
from dotenv import load_dotenv
from PIL import Image
import uuid
import pandas as pd

st.set_page_config(layout="wide")

def load_environment_variables():
    """Load environment variables from the .env file."""
    load_dotenv()
    api_key = st.secrets['OPENAI_API_KEY']
    assistant_id = st.secrets['ASSISTANT_ID']
    vector_store_id = st.secrets['VECTOR_STORE_ID']
    
    return api_key, assistant_id, vector_store_id

def initialize_openai_client(api_key):
    """Initialize the OpenAI client with the provided API key."""
    return openai.OpenAI(api_key=api_key)

def chat_with_gpt(client, vector_store_id, assistant_id, messages):
    """Interact with the GPT-4 assistant using the provided messages."""
    try:
        if vector_store_id is None or assistant_id is None:
            st.error("⚠️ No vector store or assistant available. Please make sure the environment variables are set correctly.")
            return None
        
        # Create a thread with the user's messages
        thread = client.beta.threads.create(
            messages=[
                {"role": "user", "content": msg["content"]} for msg in messages
            ]
        )

        # Run the assistant using the saved assistant ID and vector store ID
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id,
        )

        # Wait for the run to complete
        while run.status != "completed":
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        # Retrieve the assistant's response
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        assistant_response = ""
        for message in messages.data:
            if message.role == "assistant":
                for content in message.content:
                    if content.type == 'text':
                        assistant_response += content.text.value + "\n"
        return assistant_response
            
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
        return None

def display_sidebar_logo(logo_path):
    """Display the logo on the sidebar."""
    logo = Image.open(logo_path)
    st.sidebar.image(logo, use_column_width=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'chats' not in st.session_state:
        st.session_state.chats = {}
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None

def handle_chat_management():
    """Handle chat management, including creating and selecting chats."""
    st.sidebar.title("**Chats**")
    if st.sidebar.button("➕ New Chat"):
        new_chat_id = str(uuid.uuid4())
        st.session_state.chats[new_chat_id] = {
            "name": f"🗨️ Chat {len(st.session_state.chats) + 1}",
            "messages": [],
            "selected_option": None  # To keep track of the selected option per chat
        }
        st.session_state.current_chat_id = new_chat_id

    for chat_id, chat in st.session_state.chats.items():
        if st.sidebar.button(chat["name"], key=chat_id):
            st.session_state.current_chat_id = chat_id

def display_chat_interface(client, vector_store_id, assistant_id):
    """Display the main chat interface and handle user interactions."""
    
    # Title and Branding
    st.header(":green[Max]Lite :gray[- CrossOver Tool]", divider='gray')
    # st.subheader("Powered by ***Pull Logic***", divider='gray')

    if st.session_state.current_chat_id:
        current_chat = st.session_state.chats[st.session_state.current_chat_id]

        # Ensure selected_option is initialized with a default value if it's None
        if current_chat.get("selected_option") is None:
            current_chat["selected_option"] = "💬 Standard Chatbot"

        # Select option
        options = ["💬 Standard Chatbot", "📂 Upload File and Compare", "📝 Enter Specs and Compare"]
        current_chat["selected_option"] = st.selectbox(
            "Choose a function", options, index=options.index(current_chat["selected_option"])
        )

        # Display the chat history for the current chat with icons like ChatGPT
        for msg in current_chat["messages"]:
            if msg["role"] == "user":
                st.chat_message("user", avatar="🧑").markdown(f"**You:** {msg['content']}")
            elif msg["role"] == "assistant":
                st.chat_message("assistant", avatar="🤖").markdown(f"**Assistant:** {msg['content']}")

        if current_chat["selected_option"] == "💬 Standard Chatbot":
            display_standard_chatbot(client, vector_store_id, assistant_id, current_chat)
        elif current_chat["selected_option"] == "📂 Upload File and Compare":
            display_file_upload_chatbot(client, vector_store_id, assistant_id, current_chat)
        elif current_chat["selected_option"] == "📝 Enter Specs and Compare":
            display_specs_comparison_chatbot(client, vector_store_id, assistant_id, current_chat)

    else:
        st.write("Select a chat from the sidebar or create a new one to start.")

def stream_response(response_text, delay=0.01):
    message_placeholder = st.empty()
    streamed_text = ""
    
    # Loop through each chunk (character, word, or sentence)
    for chunk in response_text:
        streamed_text += chunk
        # Update the message with new content
        message_placeholder.chat_message("assistant", avatar="🤖").markdown(f"**Assistant:** {streamed_text}")
        time.sleep(delay)  # Simulate delay for the streaming effect

def display_standard_chatbot(client, vector_store_id, assistant_id, current_chat):
    """Display the standard chatbot interface with prompt functionality."""
    # Chat input
    if prompt := st.chat_input("What is your question?"):
        # Display user's input immediately
        user_message = {"role": "user", "content": prompt, "id": str(uuid.uuid4())}
        current_chat["messages"].append(user_message)
        st.chat_message("user", avatar="💡").markdown(f"**You:** {prompt}")
        
        # Handle the assistant's response
        with st.spinner('💬 Generating response...'):
            assistant_response = chat_with_gpt(client, vector_store_id, assistant_id, current_chat["messages"])

        if assistant_response:
            # Append and display assistant's response immediately
            assistant_message = {"role": "assistant", "content": assistant_response, "id": str(uuid.uuid4())}
            current_chat["messages"].append(assistant_message)
            # st.chat_message("assistant", avatar="🤖").markdown(f"**Assistant:** {assistant_response}")
            stream_response(assistant_response)
            
    # # Clear chat button
    # if st.button('🗑️ Clear Current Chat'):
    #     current_chat["messages"] = []
    #     st.experimental_rerun()

def display_file_upload_chatbot(client, vector_store_id, assistant_id, current_chat):
    """Handle file upload and comparison functionality."""
    uploaded_file = st.file_uploader("Upload a product specifications file", type=["csv"])
    # display_standard_chatbot(client, vector_store_id, assistant_id, current_chat)

    if uploaded_file:
        # Read the uploaded CSV file
        df = pd.read_csv(uploaded_file)
        st.write("📄 **Uploaded File:**", df)

        part_input = st.text_area("Enter part number you wish to cross with Maxlite")
        selected_row_index = df[df.apply(lambda row: row.astype(str).str.contains(str(part_input)).any(), axis=1)].index[0]

        # selected_row_index = st.selectbox("🔍 Select a row to compare", df.index)
        if st.button("🔍 Find a comparable product from Maxlite's offerings"):
            selected_row = df.iloc[selected_row_index]
            # Send the selected row as input to the chatbot
            comparison_input = selected_row.to_dict()
            handle_comparison(client, assistant_id, vector_store_id, current_chat, comparison_input)

def display_specs_comparison_chatbot(client, vector_store_id, assistant_id, current_chat):
    """Handle product specification input and comparison functionality."""
    specs_input = st.text_area("Enter product specifications")
    # display_standard_chatbot(client, vector_store_id, assistant_id, current_chat)

    if st.button("🔍 Find a comparable product from Maxlite's offerings"):
        handle_comparison(client, assistant_id, vector_store_id, current_chat, specs_input)


def handle_comparison(client, assistant_id, vector_store_id, current_chat, comparison_input):
    """Handle the comparison logic with Maxlite products."""
    prompt = f"Compare the following product with Maxlite products: {comparison_input}"
    
    # Create messages for the chatbot
    user_message = {"role": "user", "content": prompt, "id": str(uuid.uuid4())}
    current_chat["messages"].append(user_message)
    st.chat_message("user", avatar="💡").markdown(f"**You:** {prompt}")
    
    with st.spinner('Finding comparable Maxlite products...'):
        assistant_response = chat_with_gpt(client, vector_store_id, assistant_id, current_chat["messages"])

    if assistant_response:
        assistant_message = {"role": "assistant", "content": assistant_response, "id": str(uuid.uuid4())}
        current_chat["messages"].append(assistant_message)
        stream_response(assistant_response)
        # st.chat_message("assistant", avatar="🤖").markdown(f"**Assistant:** {assistant_response}")

def main():
    """Main function to run the Streamlit app."""
    api_key, assistant_id, vector_store_id = load_environment_variables()
    client = initialize_openai_client(api_key)
    
    display_sidebar_logo("ML.png")
    initialize_session_state()
    handle_chat_management()
    display_chat_interface(client, vector_store_id, assistant_id)

if __name__ == "__main__":
    main()
