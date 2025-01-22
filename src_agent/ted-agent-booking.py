# Agent for booking during chatting
# Date: 22, Jan 2025
# Writer: Ted, Jung
# Description: Booking agent with functiontool
#              

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from typing import Optional
from llama_index.core.tools import FunctionTool
from llama_index.core.bridge.pydantic import BaseModel

from llama_index.core.llms import ChatMessage
from llama_index.core.agent import FunctionCallingAgent

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
llm = Ollama(model="llama3.2", request_timeout=720.0)
Settings.llm = llm

# we will store booking under random IDs
bookings = {}


# we will represent and track the state of a booking as a Pydantic model
class Booking(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None


def get_booking_state(user_id: str) -> str:
    """Get the current state of a booking for a given booking ID."""
    try:
        return str(bookings[user_id].dict())
    except:
        return f"Booking ID {user_id} not found"


def update_booking(user_id: str, property: str, value: str) -> str:
    """Update a property of a booking for a given booking ID. Only enter details that are explicitly provided."""
    booking = bookings[user_id]
    setattr(booking, property, value)
    return f"Booking ID {user_id} updated with {property} = {value}"


def create_booking(user_id: str) -> str:
    """Create a new booking and return the booking ID."""
    bookings[user_id] = Booking()
    return "Booking created, but not yet confirmed. Please provide your name, email, phone, date, and time."


def confirm_booking(user_id: str) -> str:
    """Confirm a booking for a given booking ID."""
    booking = bookings[user_id]

    if booking.name is None:
        raise ValueError("Please provide your name.")

    if booking.email is None:
        raise ValueError("Please provide your email.")

    if booking.phone is None:
        raise ValueError("Please provide your phone number.")

    if booking.date is None:
        raise ValueError("Please provide the date of your booking.")

    if booking.time is None:
        raise ValueError("Please provide the time of your booking.")

    return f"Booking ID {user_id} confirmed!"


# create tools for each function
get_booking_state_tool = FunctionTool.from_defaults(fn=get_booking_state)
update_booking_tool = FunctionTool.from_defaults(fn=update_booking)
create_booking_tool = FunctionTool.from_defaults(
    fn=create_booking, return_direct=True
)
confirm_booking_tool = FunctionTool.from_defaults(
    fn=confirm_booking, return_direct=True
)

user = "John Doe"

prefix_messages = [
    ChatMessage(
        role="system",
        content=(
            f"You are now connected to the booking system and helping {user} with making a booking. "
            "Only enter details that the user has explicitly provided. "
            "Do not make up any details."
        ),
    )
]

agent = FunctionCallingAgent.from_tools(
    tools=[
        get_booking_state_tool,
        update_booking_tool,
        create_booking_tool,
        confirm_booking_tool,
    ],
    llm=llm,
    prefix_messages=prefix_messages,
    max_function_calls=10,
    allow_parallel_tool_calls=False,
    verbose=True,
)


response = agent.chat("Hello! I would like to make a booking, around 5pm?")

print(str(response))