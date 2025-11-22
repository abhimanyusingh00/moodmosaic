import torch
from torch import nn
from sentence_transformers import SentenceTransformer

class PersonalityRegressor(nn.Module):
    def __init__(self, sbert_model: str, hidden_dim: int = 256):
        super().__init__()
        self.encoder = SentenceTransformer(sbert_model)
        emb_dim = self.encoder.get_sentence_embedding_dimension()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 5),  # O, C, E, A, N
        )

    def encode(self, texts, device=None):
        """
        Get sentence embeddings and convert them to normal tensors,
        not inference tensors, so we can use them in autograd.
        """
        emb = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            device=device,
        )
        # Clone and detach to get a regular tensor (no inference_mode flag)
        return emb.detach().clone()

    def forward(self, texts, labels=None):
        # Put embeddings on same device as MLP
        device = next(self.mlp.parameters()).device
        emb = self.encode(texts, device=device)
        preds = self.mlp(emb)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(preds, labels.float())
        return {"loss": loss, "preds": preds}
