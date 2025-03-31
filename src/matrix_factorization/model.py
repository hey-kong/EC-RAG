import torch
from transformers import AutoTokenizer, AutoModel

embedding_model = "BAAI/bge-small-en-v1.5"
embedding_dim = 384

MODEL_IDS = {
    "llm": 0,
    "slm": 1,
}


class MFModel(torch.nn.Module):
    def __init__(
            self,
            dim=128,
            num_models=2,
            text_dim=embedding_dim,
            num_classes=1,
            use_proj=True,
            embedding_model_name="BAAI/bge-small-en-v1.5",
    ):
        super().__init__()
        self.use_proj = use_proj
        self.embedding_model_name = embedding_model_name

        # Model embedding matrix
        self.P = torch.nn.Embedding(num_models, dim)

        if self.use_proj:
            self.text_proj = torch.nn.Linear(text_dim, dim, bias=False)
        else:
            assert text_dim == dim, f"text_dim {text_dim} must be equal to dim {dim} if not using projection"

        self.classifier = torch.nn.Linear(dim, num_classes, bias=False)

        # Load tokenizer & model exactly as in the notebook
        self.tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(self.embedding_model_name)
        self.embedding_model.eval()  # Set to inference mode
        self.embedding_model.to(self.get_device())

    def get_device(self):
        return self.P.weight.device

    def get_prompt_embedding(self, prompt):
        inputs = self.tokenizer(
            prompt,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.get_device())

        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            last_hidden_state = outputs.last_hidden_state

        # Mean pooling over token embeddings
        prompt_embed = last_hidden_state.mean(dim=1).squeeze()

        return prompt_embed

    def forward(self, model_id, prompt):
        model_id = torch.tensor(model_id, dtype=torch.long).to(self.get_device())
        model_embed = self.P(model_id)
        model_embed = torch.nn.functional.normalize(model_embed, p=2, dim=1)
        prompt_embed = self.get_prompt_embedding(prompt)

        if self.use_proj:
            prompt_embed = self.text_proj(prompt_embed)

        return self.classifier(model_embed * prompt_embed).squeeze()

    @torch.no_grad()
    def pred_win_rate(self, prompt):
        logits = self.forward([0, 1], prompt)
        winrate = torch.sigmoid(logits[0] - logits[1]).item()
        return winrate

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
