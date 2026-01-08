import torch
import torch.nn as nn
from torch_scatter import scatter

from src_kldm.nn.embedding import FourierEmbedding, SinEmbedding
from src_kldm.nn.utils import scatter_center


class CSPVLayer(nn.Module):
    def __init__(
            self,
            dis_emb: SinEmbedding,
            hidden_dim: int = 128,
            act_fn: nn.Module = nn.SiLU(),
            ln: bool = False,
    ):
        super(CSPVLayer, self).__init__()

        self.dis_emb = dis_emb
        self.dis_dim = dis_emb.dim

        input_dim = (
                hidden_dim * 2 + 2 * self.dis_dim + 6
        )  # hidden states + distance/velocity + lattice

        self.v_proj = nn.Linear(3, dis_emb.dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )
        self.ln = ln
        if self.ln:
            self.layer_norm = nn.LayerNorm(hidden_dim)

    def edge_model(
            self,
            pos_diff: torch.Tensor,
            v: torch.Tensor,
            node_features: torch.Tensor,
            lattices: torch.Tensor,
            edge_node_index: torch.Tensor,
            edge_graph_index: torch.Tensor,
    ):

        hi, hj = node_features[edge_node_index[0]], node_features[edge_node_index[1]]
        vi, vj = v[edge_node_index[0]], v[edge_node_index[1]]
        vij = self.v_proj(vj - vi)

        pos_diff = self.dis_emb(pos_diff)
        l_edge = lattices[edge_graph_index]

        edges_input = torch.cat([hi, hj, l_edge, vij, pos_diff], dim=1)

        edge_features = self.edge_mlp(edges_input)
        return edge_features

    def node_model(
            self,
            node_features: torch.Tensor,
            edge_features: torch.Tensor,
            edge_node_index: torch.Tensor,
    ):
        agg = scatter(
            edge_features,
            edge_node_index[0],
            dim=0,
            reduce="mean",
            dim_size=node_features.shape[0],
        )
        agg = torch.cat([node_features, agg], dim=1)
        out = self.node_mlp(agg)
        return out

    def forward(
            self,
            pos_diff: torch.Tensor,
            v: torch.Tensor,
            node_features: torch.Tensor,
            l: torch.Tensor,
            edge_node_index: torch.Tensor,
            edge_graph_index: torch.Tensor,
    ):

        node_input = node_features
        if self.ln:
            node_features = self.layer_norm(node_input)

        edge_features = self.edge_model(
            pos_diff=pos_diff,
            v=v,
            node_features=node_features,
            lattices=l,
            edge_node_index=edge_node_index,
            edge_graph_index=edge_graph_index,
        )

        node_output = self.node_model(
            node_features=node_features,
            edge_features=edge_features,
            edge_node_index=edge_node_index,
        )
        return node_input + node_output


class CSPVNet(nn.Module):

    def __init__(
            self,
            hidden_dim: int = 128,
            time_dim: int = 128,
            num_layers: int = 4,
            h_dim: int = 100,
            num_freqs: int = 10,
            ln: bool = True,
            smooth: bool = False,
            pred_h: bool = False,
            pred_l: bool = True,
            pred_v: bool = True,
            zero_cog: bool = True,
            time_emb: nn.Module = None,
    ):
        super(CSPVNet, self).__init__()

        self.act_fn = nn.SiLU()

        # Embedding layers
        if smooth:
            self.node_embedding = nn.Linear(h_dim, hidden_dim, bias=False)
        else:
            # we just need to embed the given discrete h
            self.node_embedding = nn.Embedding(h_dim + 1, hidden_dim)

        self.atom_latent_emb = nn.Linear(hidden_dim + time_dim, hidden_dim)

        self.dis_emb = SinEmbedding(n_frequencies=num_freqs)

        if time_emb is None:
            time_emb = FourierEmbedding(in_features=1, out_features=time_dim)

        self.time_emb = time_emb

        # Message-passing layers
        self.layers = nn.ModuleList(
            [
                CSPVLayer(
                    self.dis_emb, hidden_dim=hidden_dim, act_fn=self.act_fn, ln=ln
                )
                for _ in range(num_layers)
            ]
        )

        # Readout layers
        if ln:
            self.final_layer_norm = nn.LayerNorm(hidden_dim)

        if pred_v:
            self.out_v = self.out_v = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.act_fn,
                nn.Linear(hidden_dim, 3, bias=False),
            )

        if pred_l:
            self.out_l = nn.Linear(hidden_dim, 6, bias=False)

        if pred_h:
            self.out_h = nn.Linear(hidden_dim, h_dim)

        self.ln = ln
        self.smooth = smooth
        self.pred_h = pred_h
        self.pred_l = pred_l
        self.pred_v = pred_v
        self.zero_cog = zero_cog

    def forward(
            self,
            t: torch.Tensor,
            pos: torch.Tensor,
            v: torch.Tensor,
            h: torch.Tensor,
            l: torch.Tensor,
            node_index: torch.Tensor,
            edge_node_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        t = self.time_emb(t)
        t_per_atom = t[node_index]

        node_features = self.node_embedding(h)
        node_features = torch.cat([node_features, t_per_atom], dim=1)
        node_features = self.atom_latent_emb(node_features)

        pos_diff = pos[edge_node_index[1]] - pos[edge_node_index[0]]
        edge_graph_index = node_index[edge_node_index[0]]

        for layer in self.layers:
            node_features = layer.forward(
                pos_diff=pos_diff,
                v=v,
                node_features=node_features,
                l=l,
                edge_node_index=edge_node_index,
                edge_graph_index=edge_graph_index,
            )

        if self.ln:
            node_features = self.final_layer_norm(node_features)

        out = dict()

        if self.pred_v:
            out_v = self.out_v(node_features)

            if self.zero_cog:
                out_v = scatter_center(out_v, index=node_index)

            out["v"] = out_v

        if self.pred_l:
            graph_features = scatter(node_features, node_index, dim=0, reduce="mean")
            out_l = self.out_l(graph_features)
            out_l = out_l.view(-1, 6)
            out["l"] = out_l

        if self.pred_h:
            out_h = self.out_h(node_features)
            out["h"] = out_h

        return out
