import os
from dotenv import load_dotenv
from huggingface_hub import login


# load Hugging Face token to access models
def load_hf_token():
    load_dotenv()
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    login(hf_token)
