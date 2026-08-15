import torch
from transformers import CLIPTokenizer, CLIPTextModelWithProjection

class QueryEmbedder:
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        """
        Initialise CLIP text encoder - base-patch32natively outputs required
        for 512-dimensional vectors.
        """
        print("Loading CLIP model... (this may take a moment the first time)")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)

        # Ensure text features match shared multimodal space
        self.model = CLIPTextModelWithProjection.from_pretrained(model_name)

        # Set evaluation mode
        self.model.eval()

    def embed_query(self, text_query: str) -> list[float]:
        """
        Convert text query into 512-dimensional normalised vector list.
        """
        # Tokenise text
        inputs = self.tokenizer(
            [text_query], 
            padding=True, 
            return_tensors="pt"
        )

        # Generate embeddings without calculating gradients
        # (saves memory)
        with torch.no_grad():
            outputs = self.model(**inputs)
            text_embeds = outputs.text_embeds
            
        # L2 Normalisation (required for cosine similarity in vector databases)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Flatten into standard Python list for ChromaDB
        return text_embeds.squeeze().tolist()


if __name__ == "__main__":
    # Quick test
    embedder = QueryEmbedder()
    test_query = "show me scenes with people talking indoors"
    
    vector = embedder.embed_query(test_query)
    print(f"\nSuccessfully embedded query: '{test_query}'")
    print(f"Vector length: {len(vector)} (Should be 512)")
    print(f"First 5 dimensions: {vector[:5]}")