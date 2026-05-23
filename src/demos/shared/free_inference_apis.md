# Using Free Inference APIs

- Step 1: Get the API keys (free) at:
    - Groq: https://console.groq.com/keys
    - Gemini: https://aistudio.google.com/app/api-keys
- Step 2: Dependencies: `pip install groq google-genai` or `uv add groq google-genai`.
- Step 3: Access the api keys in the pythonscript/notebook. One suggested approach: using `python-dotenv`
    - `pip install python-dotenv` or `uv add python-dotenv`.
    - add a file named `.env` at project root and add: 
  ```
  GROQ_API_KEY=your_groq_api_key_here
  GEMINI_API_KEY=your_gemini_api_key_here
  ```
    - Then, access the keys with:
  ```
  from dotenv import load_dotenv
  import os

  load_dotenv()
  groq_api_key = os.getenv("GROQ_API_KEY")
  gemini_api_key = os.getenv("GEMINI_API_KEY")
  ```

  - Step 5:Model inference. Refer also to free_inference_apis.ipynb
  ```
  from groq import Groq

groq_client = Groq(api_key=GROQ_API_KEY)
response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user",   "content": "Hello?"},
    ],
    temperature=0.7,
    max_tokens=300,
)

print(response.choices[0].message.content)
```

- Or with Gemini:
```
from google import genai
from google.genai import types

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
response = gemini_client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is the core idea behind Shapley values in one paragraph?",
    config=types.GenerateContentConfig(
        system_instruction="You are a concise assistant.",
        temperature=0.7,
        max_output_tokens=300,
    ),
)

print(response.text)
```
  
        
