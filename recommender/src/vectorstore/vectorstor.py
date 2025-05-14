import json
from pathlib import Path
from typing import List
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from src.base.imagedescriber import ImageDescriber, LlavaOllamaDescriber  # Assuming this is where BLIP2Describer is located


class WardrobeVectorStore:
    def __init__(self, json_file: str, vectorstore_dir: str = "vectorstore/faiss_index", image_describer: ImageDescriber = LlavaOllamaDescriber()):
        # Load wardrobe items from JSON
        self.json_file = json_file
        self.vectorstore_dir = vectorstore_dir
        self.items = self.load_items(json_file)

        # Initialize the Image Describer (BLIP2)
        self.image_describer = image_describer

        # Embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Initialize FAISS vector store
        self.vectorstore = None

    def load_items(self, json_file: str) -> List[dict]:
        """Load items from the provided JSON file."""
        with open(json_file, "r") as f:
            return json.load(f)

    def item_to_document(self, item: dict) -> Document:
        """Convert each wardrobe item to a LangChain Document."""
        # Generate image description using BLIP2
        description = self.image_describer.describe(item['image'])
        
        # Combine image description with item details
        description += f" A {item['color']} {item['type']} in {item['style']} style, "
        description += f"made of {item['material']}, suitable for {', '.join(item['occasion'])}. "
        description += f"Tags: {', '.join(item['tags'])}."
        
        return Document(
            page_content=description,
            metadata={
                "id": item["id"],
                "image": item["imagei"],
                "type": item["type"],
                "color": item["color"],
                "style": item["style"]
            }
        )

    def create_vectorstore(self):
        """Create a FAISS vector store from the wardrobe items."""
        documents = [self.item_to_document(item) for item in self.items]
        self.vectorstore = FAISS.from_documents(documents, self.embedding_model)
        self.vectorstore.save_local(self.vectorstore_dir)
        print(f"✅ Vector store saved to: {self.vectorstore_dir}")

    def load_vectorstore(self):
        """Load the FAISS vector store from the directory."""
        if Path(self.vectorstore_dir).exists():
            self.vectorstore = FAISS.load_local(self.vectorstore_dir, self.embedding_model)
            print(f"✅ Vector store loaded from: {self.vectorstore_dir}")
        else:
            print(f"⚠️ Vector store not found. Please create it first.")
