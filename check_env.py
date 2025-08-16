# check_env.py
from dotenv import load_dotenv
import os
load_dotenv()
print(f"Skrip Pengecek .env menemukan kunci -> '{os.environ.get('INTERNAL_SECRET_KEY')}'")