from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],  # Allows all origins (for development only)
    allow_credentials=True,
    allow_methods=['*'],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=['*'],  # Allows all headers
)

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro')


class InputPrompt(BaseModel):
    prompt_input: str


@app.get('/')
def root():
    return {
        'description': 'This is chatbot made usig Gemini chat model.'
    }


@app.post('/chat')
def chat_response(prompt_input: InputPrompt):
    input_prompt = prompt_input.prompt_input

    try:
        result = model.invoke(input_prompt)

        # print(result.content)
        return {
            'response': result.content
        }
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    uvicorn.run('main:app', host='127.0.0.1', port=8000, reload=True)
