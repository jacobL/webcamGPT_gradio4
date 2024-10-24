 
import numpy as np
import requests
import gradio as gr
from webcamgpt.utils import compose_payload
from webcamgpt.config import *
import google.generativeai as genai
from PIL import Image 
from openai import OpenAI

class OpanAIConnector:

    def __init__(self, api_key: str = OPENAI_API_KEY):
        if api_key is None:
            raise ValueError("OPENAI_API_KEY is not set")
        self.api_key = api_key

    # 圖文推論
    def prompt(self, image: np.ndarray, prompt: str, model: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = compose_payload(image=image, prompt=prompt, model=model)
        response = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers=headers, json=payload).json() 
        return  response['choices'][0]['message']['content']+'('+model+')'
    
    # 語音轉文字
    def transcript(self, audio, model, response_type) -> str:
        try:
            client = OpenAI(api_key=self.api_key) 
            audio_file = open(audio, "rb")
            transcriptions = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format=response_type
            )
        except Exception as error:
            print(str(error))
            raise gr.Error("An error occurred while generating speech. Please check your API key and come back try again.")

        return transcriptions
    
class GoogleConnector:

    def __init__(self, api_key: str = GOOGLE_API_KEY):
        if api_key is None:
            raise ValueError("GOOGLE_API_KEY is not set")
        self.api_key = api_key

    # 圖文推論
    def prompt(self, image: np.ndarray, prompt: str) -> str: 
        genai.configure(api_key=self.api_key)
        generation_config = genai.types.GenerationConfig(
            temperature=0.9,
            max_output_tokens=1024, 
            top_k=32,
            top_p=1,
        )
        model_name = "gemini-1.5-pro-latest"  
        model = genai.GenerativeModel(model_name)
 
        # Convert array to image
        array = np.array(image)
        image = Image.fromarray(array)
        image.save('output.png')
        inputs = [prompt,image]
        response = model.generate_content(inputs, stream=True, generation_config=generation_config)
        response.resolve()
        print(response)  
        return response.text+'(gemini-1.5-pro)'
     