import chromadb
import torch
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import os  # Added missing import

# Initialize models and DB
class MultimodalRAG:
    def __init__(self):
        # Text embedding model
        self.text_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
        # Image embedding model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # QA model
        self.qa_pipeline = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # ChromaDB client
        self.client = chromadb.PersistentClient(path="vector_db")
        self.text_collection = self.client.get_or_create_collection("Analysis_texts")
        self.image_collection = self.client.get_or_create_collection("Analysis_images")
    
    def hybrid_search(self, query_text=None, query_image=None):
        results = {}
        
        # Text search
        if query_text:
            text_emb = self.text_model.encode(query_text).tolist()
            text_results = self.text_collection.query(
                query_embeddings=[text_emb],
                n_results=3
            )
            results["text"] = text_results
        
        # Image search
        if query_image:
            if isinstance(query_image, str):  # Path to image
                img = self.clip_preprocess(Image.open(query_image)).unsqueeze(0).to(self.device)
            else:  # PIL Image object
                img = self.clip_preprocess(query_image).unsqueeze(0).to(self.device)
            
            img_emb = self.clip_model.encode_image(img).tolist()[0]
            image_results = self.image_collection.query(
                query_embeddings=[img_emb],
                n_results=3
            )
            results["images"] = image_results
        
        return results
    
    def generate_answer(self, question, context=None, image_context=None):
        # If no context provided, perform retrieval first
        search_results = None
        if context is None:
            search_results = self.hybrid_search(query_text=question)
            if search_results.get("text"):
                docs = search_results["text"]["documents"][0]
                if isinstance(docs, list):
                    context = " ".join(docs)
                else:
                    context = docs
        
        # Generate answer from text context
        if context:
            answer = self.qa_pipeline(
                question=question,
                context=context,
                handle_impossible_answer=True
            )
            # Ensure answer is not empty
            if not answer.get("answer"):
                answer["answer"] = "No relevant answer found in the provided context."
                answer["score"] = 0.0
            # Add citations if available
            if search_results and "text" in search_results:
                answer["source"] = {
                    "page": search_results["text"]["ids"][0][0],
                    "text": (search_results["text"]["documents"][0][0] if isinstance(search_results["text"]["documents"][0], list) else search_results["text"]["documents"][0]) + "..."
                }
            return answer
        # Handle image-only questions
        elif image_context:
            return {"answer": "Image analysis results would go here", "score": 1.0}
        return {"answer": "No relevant information found", "score": 0.0}
    
    def process_upload(self, file_path, file_type="pdf"):
        try:
            if file_type == "pdf":
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                
                for page_num, page in enumerate(reader.pages):
                    # Process text
                    text = page.extract_text()
                    if text:
                        text_emb = self.text_model.encode(text).tolist()
                        self.text_collection.add(
                            documents=[text],
                            embeddings=[text_emb],
                            ids=[f"text_{page_num}_{os.path.basename(file_path)}"]
                        )
                    
                    # Process images
                    for img_num, img in enumerate(page.images):
                        temp_img_path = f"temp_img_{page_num}_{img_num}.jpg"
                        with open(temp_img_path, "wb") as f:
                            f.write(img.data)
                        
                        try:
                            image = self.clip_preprocess(Image.open(temp_img_path)).unsqueeze(0).to(self.device)
                            image_emb = self.clip_model.encode_image(image).tolist()[0]
                            self.image_collection.add(
                                embeddings=[image_emb],
                                ids=[f"img_{page_num}_{img_num}_{os.path.basename(file_path)}"]
                            )
                        finally:
                            if os.path.exists(temp_img_path):
                                os.remove(temp_img_path)
                
                return True
                
        except Exception as e:  # Fixed syntax error here
            print(f"Error processing {file_path}: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    rag = MultimodalRAG()
    
    # Example QA
    question = "What were the key findings about Instagram usage in the study?"
    answer = rag.generate_answer(question)
    print(f"Question: {question}")
    print(f"Answer: {answer['answer']}")
    print(f"Confidence: {answer['score']:.2f}")
    if "source" in answer:
        print(f"Source: {answer['source']['text']}")