import torch
import torch.nn as nn
import torch.nn.functional as F


class PerAttrCondEmbedder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.parents = args.parents
        self.parent_dims = args.parent_dims
        self.cond_embed_dim = args.cond_embed_dim

        assert self.cond_embed_dim % len(self.parents) == 0, (
            f"cond_embed_dim ({self.cond_embed_dim}) must be divisible by "
            f"number of parents ({len(self.parents)})"
        )

        self.parent_embed_dim = self.cond_embed_dim // len(self.parents)

        self.embeddings = nn.ModuleDict()
        self.null_embeddings = nn.ParameterDict()

        for key in self.parents:
            in_dim = self.parent_dims[key]
            self.embeddings[key] = nn.Sequential(
                nn.Linear(in_dim, self.parent_embed_dim),
                nn.SiLU(),
                nn.Linear(self.parent_embed_dim, self.parent_embed_dim),
            )
            self.null_embeddings[key] = nn.Parameter(
                torch.zeros(1, self.parent_embed_dim)
            )

    def forward(self, y_dict, null_keys=None):
        null_keys = set() if null_keys is None else set(null_keys)
        embs = []

        for key in self.parents:
            assert key in y_dict, f"Missing key in y_dict: {key}"
            val = y_dict[key]  # [B, in_dim]

            if key in null_keys:
                emb = self.null_embeddings[key].expand(val.shape[0], -1)
            else:
                emb = self.embeddings[key](val)

            embs.append(emb)

        out = torch.cat(embs, dim=1)   # [B, cond_embed_dim]
        out = F.normalize(out, p=2, dim=1)
        return out


class GlobalCondEmbedder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.parents = args.parents
        self.parent_dims = args.parent_dims
        self.cond_embed_dim = args.cond_embed_dim

        self.input_dim = sum(self.parent_dims[k] for k in self.parents)

        self.embed = nn.Sequential(
            nn.Linear(self.input_dim, self.cond_embed_dim),
            nn.SiLU(),
            nn.Linear(self.cond_embed_dim, self.cond_embed_dim),
        )

    def forward(self, y_dict):
        vals = []
        for key in self.parents:
            assert key in y_dict, f"Missing key in y_dict: {key}"
            vals.append(y_dict[key])

        concat_y = torch.cat(vals, dim=1)
        out = self.embed(concat_y)
        out = F.normalize(out, p=2, dim=1)
        return out