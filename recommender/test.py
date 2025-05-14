from src.vectorstore.vectorstor import WardrobeVectorStore

store = WardrobeVectorStore("data/data.json", "vectorstore/faiss_index")

#store.create_vectorstore()

store.load_vectorstore(allow_dangerous_deserialization=True)

print(store.vectorstore.similarity_search("a yellow t-shirt"))
