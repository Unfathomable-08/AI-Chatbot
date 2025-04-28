import chainlit as cl
from cosine_sim import compute_cosine_similarity

def preprocess_text(text):
    """Remove extra spaces before punctuation marks."""
    text = text.replace(" ,", ",").replace(" ?", "?").replace(" !", "!").replace(" .", ".")
    return text.strip()

@cl.on_message
async def main(message: cl.Message):
    try:
        # Extract user input from the message
        input_data = message.content.strip()

        # Call the cosine similarity
        response = compute_cosine_similarity(input_data)

        # Remove extra spaces beforepanctuations
        response = preprocess_text(response)

        # Send the prediction back to the user
        await cl.Message(content=response).send()

    except Exception as e:
        # Handle errors
        await cl.Message(content=f"Error: {str(e)}. Please check your input and try again.").send()