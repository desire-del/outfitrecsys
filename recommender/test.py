from src.base.imagedescriber import BLIP2Describer, LlavaOllamaDescriber

image_path="data/images/image1.png"
prompt = """You are a fashion assistant. Look at the image and describe the outfit in detail.
 Mention the type of clothing, color, material, and style (e.g. casual, formal, sporty).
   Also suggest occasions where it can be worn (e.g. office, beach, party).
     Keep the description under 50 words and use fashion vocabulary."""

descp = LlavaOllamaDescriber()
print(descp.describe(image_path, prompt))

