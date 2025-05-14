from abc import ABC, abstractmethod
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from transformers import pipeline
from typing import Optional
import base64
import requests
import os



class ImageDescriber(ABC):
    @abstractmethod
    def describe(self, image_path: str, **kwargs) -> str:
        """Return a textual description of the image."""
        pass




class BLIP2Describer:
    def __init__(
        self, 
        model_name: str = "Salesforce/blip2-opt-2.7b", 
        device: Optional[str] = None,
        max_new_tokens: int = 50,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_8bit: bool = False
    ):
        """
        BLIP-2 Image Describer
        
        Args:
            model_name: BLIP-2 model name
            device: Device to run on (None for auto)
            max_new_tokens: Max length of generated description
            torch_dtype: Precision (None for auto)
            load_in_8bit: Use 8-bit quantization to save memory
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if "cuda" in self.device else torch.float32)
        self.max_new_tokens = max_new_tokens
        
        # Initialize processor and model with proper device handling
        self.processor = Blip2Processor.from_pretrained(model_name)
        
        # Special handling for different devices
        if "cuda" in self.device:
            kwargs = {
                "device_map": "auto",
                "torch_dtype": self.torch_dtype,
            }
            if load_in_8bit:
                kwargs["load_in_8bit"] = True
        else:
            kwargs = {"device_map": {"": self.device}}
            
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            **kwargs
        )
        
        # Disable tokenizer parallelism
        self.processor.tokenizer.padding_side = "left"

    def describe(
        self, 
        image_path: str, 
        prompt: Optional[str] = None
    ) -> str:
        """
        Generate description for an image
        
        Args:
            image_path: Path to image file
            prompt: Optional prompt/question
            
        Returns:
            Generated description
        """
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Process inputs - let processor handle device placement
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt"
            ).to(self.model.device)
            
            # Generate description
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            
            return self.processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0].strip()
            
        except Exception as e:
            raise RuntimeError(f"Error processing image {image_path}: {str(e)}")

    def __del__(self):
        """Clean up when instance is deleted"""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LlavaOllamaDescriber(ImageDescriber):
    def __init__(self, model_name: str = "llava:7b"):
        self.model_name = model_name
        self.endpoint = "http://localhost:11434/api/generate"

    def _image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def describe(self, image_path: str, prompt: str) -> str:
        image_b64 = self._image_to_base64(image_path)
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False
        }

        response = requests.post(self.endpoint, json=payload)
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            raise RuntimeError(f"Failed to get response from Ollama: {response.text}")
