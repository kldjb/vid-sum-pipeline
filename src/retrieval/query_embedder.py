import torch
import clip


class QueryEmbedder:
    def __init__(
            self,
            model_name: str = "ViT-B/16",
            weights_path: str = "model/vt_clip.pth",
            device: str = None,
        ):
        """
        Initialise native OpenAI CLIP model and inject fine-tuned VT-CLIP weights.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load CLIP with jit set to False to load custom state dicts cleanly
        print(f"Loading native CLIP model ({model_name}) on {self.device}...")
        self.model, self.preprocess = clip.load(
            model_name,
            device=self.device,
            jit=False,
        )

        print(f"Injecting custom VT-CLIP weights from {weights_path}...")
        try:
            checkpoint = torch.load(
                weights_path,
                map_location=self.device,
                weights_only=False,
            )

            # Extract state dict if wrapped in training metadata
            if isinstance(checkpoint, dict):
                custom_state_dict = checkpoint.get(
                    "state_dict",
                    checkpoint.get("model", checkpoint)
                )
            else:
                custom_state_dict = checkpoint

            # Strip prefixes
            cleaned_state_dict = {}
            for k, v in custom_state_dict.items():
                new_key = k
                for prefix in ("module.", "clip_model.", "model."):
                    if new_key.startswith(prefix):
                        new_key = new_key[len(prefix):]
                cleaned_state_dict[new_key] = v

            # Load weights
            incompatible_keys = self.model.load_state_dict(
                cleaned_state_dict,
                strict=False,
            )
            
            missing = len(incompatible_keys.missing_keys)
            unexpected = len(incompatible_keys.unexpected_keys)
            print(
                f"Successfully loaded VT-CLIP weights! (Missing keys: {missing},"
                 f" Unexpected keys: {unexpected})"
                )

        except FileNotFoundError:
            print(
                f"WARNING: Could not find {weights_path}. "
                "Cosine similarities will be misaligned!"
            )

        self.model.eval()

    def embed_query(self, text_query: str) -> list[float]:
        """
        Convert text query into 512-dimensional normalised vector list.
        """
        # Tokenise (truncates automatically if > 77 tokens)
        text_tokens = clip.tokenize([text_query], truncate=True).to(self.device)

        with torch.no_grad():
            # get text features from model
            text_features = self.model.encode_text(text_tokens)

            # L2 normalise for cosine similarity in vector database
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features.squeeze().cpu().tolist()


if __name__ == "__main__":
    embedder = QueryEmbedder()
    test_query = "show me scenes with people talking indoors"

    vector = embedder.embed_query(test_query)
    print(f"\nSuccessfully embedded query: '{test_query}'")
    print(f"Vector length: {len(vector)} (Should be 512)")
    print(f"First 5 dimensions: {vector[:5]}")