import base64
import requests

class LMStudioVLMClient:
    def __init__(self, base_url="http://localhost:1234/v1", model=None):
        self.base_url = base_url.rstrip("/")
        # Jika model tidak diisi, gunakan default model LLaVA/Qwen2-VL Anda
        self.model = model if model else "qwen2-vl-2b-instruct"

    def _encode_image(self, image_path):
        """Membaca file gambar dan mengonversinya ke Base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def predict_plate(self, image_path, prompt="What is the license plate number shown in this image? Respond only with the plate number."):
        base64_image = self._encode_image(image_path)
        
        # URL Endpoint OpenAI-compatible untuk chat completions
        url = f"{self.base_url}/chat/completions"

        # Tentukan tipe mime berdasarkan ekstensi
        ext = image_path.lower().split('.')[-1]
        mime_type = "image/png" if ext == "png" else "image/jpeg"

        payload = {
            "model": self.model,
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
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 50
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(url, json=payload, headers=headers, timeout=180)
        
        if response.status_code != 200:
            raise Exception(f"{response.status_code} Client Error: {response.text}")

        data = response.json()
        
        # Ekstrak teks hasil respon model
        prediction = data["choices"][0]["message"]["content"].strip()
        return prediction