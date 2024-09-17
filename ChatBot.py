import streamlit as st
import time
import openai
import os
from dotenv import load_dotenv
from PIL import Image
import uuid
import pandas as pd

st.set_page_config(layout="wide")
# if 'current_question' not in st.session_state:
#     st.session_state.current_question = 1  # Start with the first question

def load_environment_variables():
    """Load environment variables from the .env file."""
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    assistant_id = os.getenv('ASSISTANT_ID')
    vector_store_id = os.getenv('VECTOR_STORE_ID')
    
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

def stream_response(response_text, delay=0.01):
    message_placeholder = st.empty()
    streamed_text = ""
    
    # Loop through each chunk (character, word, or sentence)
    for chunk in response_text:
        streamed_text += chunk
        # Update the message with new content
        message_placeholder.chat_message("assistant", avatar="🤖").markdown(f"**Assistant:** {streamed_text}")
        time.sleep(delay)  # Simulate delay for the streaming effect

def transform_chat(user_message):
    if 'The user needs' in user_message:
        button_message = 'Check Product Availability'
    
    elif 'Please provide the product type and features of the Maxlite product with catalog number' in user_message:
        button_message = 'Get alternatives to input Maxlite product'
    
    else: 
        button_message = ''

    return button_message

def display_chat_interface(client, vector_store_id, assistant_id):
    """Display the main chat interface and handle user interactions."""
    
    # Title and Branding
    st.header(":green[Max]Lite :gray[- CrossOver Tool]", divider='gray')
    # st.subheader("Powered by ***Pull Logic***", divider='gray')

    if st.session_state.current_chat_id:
        current_chat = st.session_state.chats[st.session_state.current_chat_id]

        # Chatbot options
        options = ["Get inventory availability information",
                   "Get alternatives to other manufacturer's products", 
                   "Get alternatives to Maxlite products",
                   "Get product recommendations based on requirement", 
                   ]

        # Ensure selected_option is initialized with a default value if it's None
        if current_chat.get("selected_option") is None:
            current_chat["selected_option"] = options[0]

        # Select option
        
        current_chat["selected_option"] = st.selectbox(
            "Choose a function", options, index=options.index(current_chat["selected_option"])
        )

        # Display the chat history for the current chat with icons like ChatGPT
        for msg in current_chat["messages"]:

            if msg["role"] == "user":
                transformed_msg = transform_chat(msg['content'])
                if transformed_msg != '':
                    st.chat_message("user", avatar="💡").markdown(f"**You:** {transformed_msg}")
            elif msg["role"] == "assistant":
                st.chat_message("assistant", avatar="🤖").markdown(f"**Assistant:** {msg['content']}")

        if current_chat["selected_option"] == "Get alternatives to Maxlite products":
            maxlite_alternatives(client, vector_store_id, assistant_id, current_chat)
        elif current_chat["selected_option"] == "Get alternatives to other manufacturer's products":
            competitor_alternatives(client, vector_store_id, assistant_id, current_chat)
        elif current_chat["selected_option"] == "Get inventory availability information":
            get_availability(client, vector_store_id, assistant_id, current_chat)
        elif current_chat["selected_option"] == "Get product recommendations based on requirement":
            get_recommendations(client, vector_store_id, assistant_id, current_chat)

    else:
        st.write("Select a chat from the sidebar or create a new one to start.")

def maxlite_alternatives(client, vector_store_id, assistant_id, current_chat):
    # Ensure the question state persists
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0

    if st.session_state.current_question == 0:
        ml_input1 = st.text_input("Enter Catalog Number for Maxlite product")
        
        if ml_input1:  # Only proceed if the user has provided input
            prompt1 = f'Please provide the product type and features of the Maxlite product with catalog number {ml_input1}. Do not mention price of the product. Just provide a summary at the end and do not ask further questions.'
            # get_response(client, assistant_id, vector_store_id, current_chat, prompt1, 'Getting product details...')
        
            prompt2 = f'Compare the following product with Maxlite products: Maxlite product with catalog number {ml_input1}. Do not mention price of the alternatives. Also, based on the similarity of features, please add a final row for "Substitution Fitness Score" which is a score between 0-100%'
            get_response(client, assistant_id, vector_store_id, current_chat, prompt2, 'Finding comparable Maxlite products...')
            st.session_state.current_question = 1  # Move to the next step

    if st.session_state.current_question == 1:
        if st.button("Get Inventory Availability"):
            st.session_state.current_question = 2  # Move to the next step

    if st.session_state.current_question == 2:
        product_list = get_products(client, assistant_id, vector_store_id, current_chat)
        product_list = product_list.split('"')[1:-1]
        p_list = [p for p in product_list if "," not in p]

        with st.form(key='inventory_form'):
            c1, c2, c3, c4 = st.columns(4)
            
            product = c1.radio(label='Select Product', options=p_list, key='product_selection')            
            qty = c2.text_input(label='Required Quantity', key='qty')
            date_needed = c3.date_input(label="Required Delivery Date", key='date_needed')
            zip_cd = c4.text_input(label='Delivery ZIP Code', key='zip_cd')

            submit = st.form_submit_button("Submit")

            if submit:
                prompt1 = f'The user needs {qty} units of product {product} on {date_needed} at {zip_cd}. Provide a short answer saying Right now we have enough on-hand for this opportunity if the order is placed with the next 24 hours. We expect you to receive the product within x business days of the product shipping. Please submit the purchase order to your Maxlite sales representative. Fill x as a random integer between 3 and 6 days.'
                get_response(client, assistant_id, vector_store_id, current_chat, prompt1, 'Getting product availability...')

def competitor_alternatives(client, vector_store_id, assistant_id, current_chat):
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0

    with st.form(key='competitor'):
        c1, c2 = st.columns(2)
        
        manufacturer = c1.text_input(label='Manufacturer')
        cat_num = c2.text_input(label='Catalog Number')
        submit = st.form_submit_button("Find comparable Maxlite products")

        if submit:
            prompt1 = f"Please provide the product type and features of the manufacturer {manufacturer}'s product with catalog number {cat_num}. Just provide a summary at the end and do not ask further questions."
            # get_response(client, assistant_id, vector_store_id, current_chat, prompt1, 'Getting product details...')
        
            prompt2 = f"Compare the following product with Maxlite products: manufacturer {manufacturer}'s product with catalog number {cat_num}. Provide alternatives with similar wattage, lumen output, and voltage. Also, based on the similarity of features, please add a final row in the comparison table for 'Substitution Fitness Score' which is a score between 0-100%. Do not mention price of the alternatives."
            get_response(client, assistant_id, vector_store_id, current_chat, prompt2, 'Finding comparable Maxlite products...')
            st.session_state.current_question = 1  # Move to the next step

    if st.session_state.current_question == 1:
        st.button('Explore more alternatives')
        st.button('Check Product Availability', key='competitor_availability')

def get_recommendations(client, vector_store_id, assistant_id, current_chat):
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0

    with st.form(key='recommendations'):
        c1, c2, c3 = st.columns(3)
        
        p_type = c1.selectbox(label='Product Type', options=['Area Lights', 'High Bays', 'Panels and Troffers', 'Retrofit Kits'])
        in_type = c2.selectbox(label='Specification Type', options=['Wattage', 'Lumens'])
        in_num = c3.text_input(label='Specification')
        submit = st.form_submit_button("Explore Maxlite products")

        if submit:
            prompt1 = f"Please provide a comparison of the various distinct {p_type} offered by Maxlite with {in_num} {in_type} specifications. Please restrict results to 20% deviation from specification. Do not include price information"
            get_response(client, assistant_id, vector_store_id, current_chat, prompt1, 'Getting product details...')
        
            st.session_state.current_question = 1  # Move to the next step

    if st.session_state.current_question == 1:
        st.button('Explore more alternatives')
        st.button('Check Product Availability', key='recommendation_availability')

def get_availability(client, vector_store_id, assistant_id, current_chat):
    with st.form(key='inventory_form'):
        c1, c2, c3, c4 = st.columns(4)
        
        product = c1.text_input(label='Product Catalog Number', key='product_selection') #, value='HL-090HF-50'
        qty = c2.text_input(label='Required Quantity', key='qty')
        date_needed = c3.date_input(label="Required Delivery Date", key='date_needed')
        zip_cd = c4.text_input(label='Delivery ZIP Code', key='zip_cd')

        submit = st.form_submit_button("Check Product Availability")

        if submit:
            prompt1 = f'The user needs {qty} units of product {product} on {date_needed} at {zip_cd}. Provide a short answer saying Right now we have enough on-hand for this opportunity if the order is placed with the next 24 hours. We expect you to receive the product within x business days of the product shipping. Please submit the purchase order to your Maxlite sales representative. Fill x as a random integer between 3 and 6 days.'
            get_response(client, assistant_id, vector_store_id, current_chat, prompt1, 'Getting product availability...')

def get_products(client, assistant_id, vector_store_id, current_chat):    
    # Create messages for the chatbot
    prompt = 'Give me a just the python list of the catalog numbers of the products recommended until now'
    user_message = {"role": "user", "content": prompt, "id": str(uuid.uuid4())}
    # current_chat["messages"].append(user_message)
    
    with st.spinner():
        assistant_response = chat_with_gpt(client, vector_store_id, assistant_id, current_chat["messages"])
    return assistant_response

def get_response(client, assistant_id, vector_store_id, current_chat, prompt, spinner_text=None):    
    # Create messages for the chatbot
    user_message = {"role": "user", "content": prompt, "id": str(uuid.uuid4())}
    current_chat["messages"].append(user_message)
    with st.spinner(spinner_text):
        assistant_response = chat_with_gpt(client, vector_store_id, assistant_id, current_chat["messages"])

    if assistant_response:
        assistant_message = {"role": "assistant", "content": assistant_response, "id": str(uuid.uuid4())}
        current_chat["messages"].append(assistant_message)
        stream_response(assistant_response)

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
