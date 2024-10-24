import cv2
import base64
import numpy as np

# 此為openai 專用
def encode_image_to_base64(image: np.ndarray) -> str:

    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Could not encode image to JPEG format.")

    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image

def compose_payload(image: np.ndarray, prompt: str, model: str) -> dict:
    base64_image = encode_image_to_base64(image)
    return {
        "model": model, #"gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
