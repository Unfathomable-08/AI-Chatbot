import chainlit as cl
from cosine_sim import compute_cosine_similarity

@cl.on_message
async def main(message: cl.Message):
    try:
        # Extract user input from the message
        input_data = message.content.strip()

        # Call the cosine similarity
        response = compute_cosine_similarity(input_data)

        # Send the prediction back to the user
        await cl.Message(content=response).send()

    except Exception as e:
        # Handle errors
        await cl.Message(content=f"Error: {str(e)}. Please check your input and try again.").send()