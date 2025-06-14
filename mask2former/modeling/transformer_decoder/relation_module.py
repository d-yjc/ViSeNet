import clip
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from ...cat_rel_dict import rel_dict

###########################################################################
# GAT:
# A GAT implemention inspired by Gordic's implementation: https://github.com/gordicaleksa/pytorch-GAT
# In reality, this is an [E]GAT since our network contains edge features.
###########################################################################
class GAT(nn.Module):
    """
    A GAT network that stacks multiple GraphAttentionLayer modules.
    
    Args:
        num_layers: number of GAT layers (here, 2)
        nheads: list of number of heads per layer (e.g. [nhead, nhead])
        nfeatures: list of feature dimensions per layer (e.g. [hidden_dim, hidden_dim, hidden_dim])
        dropout: dropout probability
    """
    def __init__(self, num_layers: int, nheads: list[int], nfeatures: list[int], dropout: float = 0.0):
        super().__init__()
        assert num_layers == len(nheads) == len(nfeatures) - 1, "Architecture parameters mismatch."
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                GraphAttentionLayer(
                    in_channels=nfeatures[i],
                    out_channels=nfeatures[i+1],
                    nhead=nheads[i],
                    dropout=dropout
                )
            )
    
    def forward(self, x: Tensor, adj: Tensor, edge_features: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x, adj, edge_features)
        return x
    
#########################################
# Helper: Dense to Edge Index Converter #
#########################################
def dense_to_edge_index(adj: Tensor) -> Tensor:
    """
    Converts a dense adjacency matrix (num_nodes x num_nodes)
    into an edge_index tensor of shape [2, num_edges] where each
    column is [source, target].
    """
    edge_index = torch.nonzero(adj, as_tuple=False).t().contiguous()
    return edge_index

##########################################################################
# Multi-head Edge-Aware Graph Attention Layer        
##########################################################################
class GraphAttentionLayer(nn.Module):
    """
    A graph attention layer that incorporates edge features into the
    attention mechanism using multi-head attention.
    
    - Node features are the instance features.
    - Edge features are the CLIP-based textual embeddings of the predicate.
    
    For each edge (i, j) and for each head h, the raw attention score is:
    
      e_{ij}^h = LeakyReLU( (W_node(x_i))^h · a_src^h + (W_node(x_j))^h · a_tgt^h 
                             + (W_edge(e_{ij}))^h summed over channels * a_edge^h )
    
    The scores are normalized over the incoming edges per target node.
    The outputs from all heads are concatenated to form the final output.
    """
    def __init__(self, in_channels: int, out_channels: int, nhead: int = 1, dropout: float = 0.0, alpha: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nhead = nhead
        # Ensure out_channels is divisible by nhead
        assert out_channels % nhead == 0, "out_channels must be divisible by nhead"
        self.head_dim = out_channels // nhead
        
        # Linear projection for node features: project to nhead * head_dim
        self.linear_proj = nn.Linear(in_channels, nhead * self.head_dim, bias=False)
        # Linear projection for edge features (assumes edge feature dim = in_channels)
        self.edge_proj = nn.Linear(in_channels, nhead * self.head_dim, bias=False)
        # Learnable attention parameters per head
        self.a_src = nn.Parameter(torch.Tensor(nhead, self.head_dim))
        self.a_tgt = nn.Parameter(torch.Tensor(nhead, self.head_dim))
        # Learnable scalar weight per head for edge features
        self.a_edge = nn.Parameter(torch.Tensor(nhead))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout) # removing dropout as per avishek's advice. -daniel 13 apr
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_tgt)
        nn.init.constant_(self.a_edge, 1.0)
    
    def forward(self, x: Tensor, adj: Tensor, edge_features: Tensor) -> Tensor:
        """
        Args:
            x: Node features [num_nodes, batch_size, in_channels]
            adj: Dense binary adjacency matrix [num_nodes, num_nodes]
            edge_features: [num_nodes, num_nodes, in_channels]
        Returns:
            Updated node features [num_nodes, batch_size, out_channels]
        """
        num_nodes, batch_size, _ = x.size()
        device = x.device
        # Convert dense adj to edge index [2, E]
        edge_index = dense_to_edge_index(adj)
        src_idx, trg_idx = edge_index  # each: [E]
        outputs = []
        for b in range(batch_size):
            node_features = x[:, b, :]  # [num_nodes, in_channels]
            # Linear projection and dropout
            h = self.linear_proj(node_features)  # [num_nodes, nhead*head_dim]
            # h = self.dropout(h)
            # Reshape to separate heads: [num_nodes, nhead, head_dim]
            h = h.view(num_nodes, self.nhead, self.head_dim)
            
            # Gather source and target node projections for each edge
            h_src = h[src_idx]  # [E, nhead, head_dim]
            h_tgt = h[trg_idx]  # [E, nhead, head_dim]
            
            # Compute per-head scores for nodes:
            src_scores = (h_src * self.a_src).sum(dim=-1)  # [E, nhead]
            tgt_scores = (h_tgt * self.a_tgt).sum(dim=-1)    # [E, nhead]
            
            # Process edge features: project and reshape
            e_feats = edge_features[src_idx, trg_idx]  # [E, in_channels]
            proj_edge = self.edge_proj(e_feats)         # [E, nhead*head_dim]
            proj_edge = proj_edge.view(-1, self.nhead, self.head_dim)  # [E, nhead, head_dim]
            # Reduce edge features to a scalar per head (sum over head_dim) and weight it
            edge_scores = proj_edge.sum(dim=-1) * self.a_edge  # [E, nhead]
            
            # Sum scores and apply LeakyReLU: [E, nhead]
            scores = self.leakyrelu(src_scores + tgt_scores + edge_scores)
            exp_scores = torch.exp(scores)
            
            # Compute denominator for softmax per target node per head
            denom = torch.zeros(num_nodes, self.nhead, device=device)
            for head in range(self.nhead):
                denom[:, head] = denom[:, head].scatter_add(0, trg_idx, exp_scores[:, head])
            # Compute attention coefficients for each edge: [E, nhead]
            attn = exp_scores / (denom[trg_idx] + 1e-16)
            # attn = self.dropout(attn)
            
            # Aggregate messages: for each head, sum source node projections weighted by attn to each target
            out = torch.zeros(num_nodes, self.nhead, self.head_dim, device=device)
            for head in range(self.nhead):
                weighted = h_src[:, head] * attn[:, head].unsqueeze(-1)  # [E, head_dim] per edge
                out[:, head] = out[:, head].scatter_add(0, trg_idx.unsqueeze(-1).expand(-1, self.head_dim), weighted)
            # Concatenate heads: [num_nodes, nhead * head_dim] which equals [num_nodes, out_channels]
            out = out.view(num_nodes, self.nhead * self.head_dim)
            outputs.append(out.unsqueeze(1))  # [num_nodes, 1, out_channels]
        out_final = torch.cat(outputs, dim=1)  # [num_nodes, batch_size, out_channels]
        return out_final

############################################################
# Build Object Edge Features Helper Function               #
############################################################
def build_object_edge_features(queries: Tensor, 
                               relations: Tensor, 
                               sbj_ids: list, 
                               obj_ids: list, 
                               pred_ids: list, 
                               text_encoder) -> tuple[Tensor, Tensor]:
    """
    Constructs:
      - A dense binary adjacency matrix [num_objects, num_objects].
      - An edge features tensor [num_objects, num_objects, hidden_dim],
        where each nonzero entry is the normalized CLIP-based embedding of the predicate.
    
    Args:
        queries: [num_objects, batch_size, hidden_dim]
        relations: [num_relations, 3] (each relation: [subject, object, predicate])
        sbj_ids, obj_ids, pred_ids: lists of IDs (one per relation)
    Returns:
        adj: [num_objects, num_objects] binary adjacency matrix.
        edge_features: [num_objects, num_objects, hidden_dim] tensor.
    """
    num_objects = queries.size(0)
    device = queries.device
    adj = torch.zeros(num_objects, num_objects, device=device)
    hidden_dim = queries.size(-1)
    edge_features = torch.zeros(num_objects, num_objects, hidden_dim, device=device)
    
    for idx, rel in enumerate(relations):
        sbj = int(rel[0].item()) if isinstance(rel[0], torch.Tensor) else int(rel[0])
        obj = int(rel[1].item()) if isinstance(rel[1], torch.Tensor) else int(rel[1])
        adj[sbj, obj] = 1.0
        predicate_text = [rel_dict.get(int(pred_ids[idx]))]
        pred_embedding = text_encoder(predicate_text)  # [1, hidden_dim]
        edge_features[sbj, obj] = pred_embedding.squeeze(0)
    return adj, edge_features

############################################################
# RelationshipModule:
# Build the object-level EGAT and apply updates to features.                           
############################################################
class RelationshipModule(nn.Module):
    def __init__(self, hidden_dim, nhead, dropout=0.0, normalize_before=False, clip_model_name="ViT-B/32", device=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.normalize_before = normalize_before
        self.clip_model, _ = clip.load(
                                    clip_model_name, 
                                    device=device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
                                    )
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Instantiate GAT.
        self.object_gat_net = GAT(
            num_layers=2, # We utilise two layers.
            nheads=[nhead, nhead],
            nfeatures=[hidden_dim, hidden_dim, hidden_dim],
            dropout=dropout
        )
    
    def encode_rel(self, texts: list) -> Tensor:
        tokens = clip.tokenize(texts).to(next(self.parameters()).device)
        with torch.no_grad():
            text_embeddings = self.clip_model.encode_text(tokens)
        return text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
    
    
    def normalize_adjacency(self, adjacency: Tensor, eps: float = 1e-6) -> Tensor:
        row_sum = adjacency.sum(dim=1, keepdim=True).clamp_min(eps)
        return adjacency / row_sum
    
    def forward(self, 
                queries: Tensor, 
                relations: Tensor, 
                sbj_ids: list = None, 
                obj_ids: list = None, 
                pred_ids: list = None) -> Tensor:
        """
        Args:
            queries: [num_objects, batch_size, hidden_dim]
            relations: [NumRelations, 3]
            sbj_ids, obj_ids, pred_ids: lists corresponding to each relation.
        Returns:
            Updated object-level queries [num_objects, batch_size, hidden_dim]
        """
        if relations is None or relations.size(0) == 0:
            return queries

        # Build object-level graph with edge features.
        adj_obj, edge_feats = build_object_edge_features(queries, relations, sbj_ids, obj_ids, pred_ids, self.encode_rel)
        # adj_obj = self.normalize_adjacency(adj_obj)

        # Apply the GAT network to update instance features.
        obj_gat_out = self.object_gat_net(queries, adj_obj, edge_feats)
        fused_queries = queries + self.norm(obj_gat_out)
        fused_queries = F.relu(fused_queries) # adding relu

        return fused_queries
