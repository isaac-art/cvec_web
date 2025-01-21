import os

from huggingface_hub import login, whoami

try: from settings_admin import *
except ImportError: from settings import *

from main import training_worker

os.environ['HF_HOME'] = HF_HOME
login(HF_KEY)
user_info = whoami()
print(f"Logged in as: {user_info['name']} ({user_info['email']})")


print("Starting worker")
training_worker()