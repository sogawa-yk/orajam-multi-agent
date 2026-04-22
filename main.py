import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OCI_GENERATIVE_AI_API_KEY")

if not api_key:
    raise ValueError("OCI_GENERATIVE_AI_API_KEY is not set. Add it to your .env file.")

client = OpenAI(
    base_url="https://inference.generativeai.ap-osaka-1.oci.oraclecloud.com/openai/v1",
    api_key=api_key,
    project="***REMOVED***"
)

response = client.responses.create(
    model="openai.gpt-4.1",
    input="What is 2x2?"
)

print(response.output_text) # should output a string like "2 x 2 = **4**."
