from flax import linen as nn
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.experimental import sparse


INF = 1e8


def normalize(x, p=2, dim=1, eps=1e-12):
    """JAX equivalent of torch.nn.functional.normalize
    
    Args:
        x: Input tensor
        p: Power for the normalization (default: 2)
        dim: Dimension to normalize over (default: 1) 
        eps: Small value to avoid division by zero (default: 1e-12)
    """
    norm = jnp.linalg.norm(x, ord=p, axis=dim, keepdims=True)
    norm = jnp.maximum(norm, eps)
    return x / norm


def laplace_norm(mat):
    # mat: sp.coo_matrix
    norm1 = sp.diags(1 / (np.sqrt(mat.sum(axis=1).A.ravel()) + 1e-8))
    norm2 = sp.diags(1 / (np.sqrt(mat.sum(axis=0).A.ravel()) + 1e-8))
    norm_mat = norm1 @ mat @ norm2
    return norm_mat


def scaled_dot_product(q, k, v, mask=None):
    dim = q.shape[-1]
    attn = jnp.matmul(q, k.swapaxes(-1, -2)) / dim ** -0.5
    if mask is not None:
        attn = jnp.where(mask == 0, -INF, attn)

    attn = nn.softmax(attn, axis=-1)
    out = jnp.matmul(attn, v)
    return out, attn


class LinNorm(nn.Module):
    n_dim: int

    def setup(self):
        self.lin1 = nn.Dense(self.n_dim * 4,
                             kernel_init=nn.initializers.xavier_uniform(),
                             bias_init=nn.initializers.zeros)

        self.act = nn.relu
        self.lin2 = nn.Dense(self.n_dim,
                             kernel_init=nn.initializers.xavier_uniform(),
                             bias_init=nn.initializers.zeros)

        self.layer_norm = nn.LayerNorm()

    def __call__(self, X):
        out = self.lin1(X)
        out = self.act(out)
        out = self.lin2(out) + X
        out = self.layer_norm(out)
        return out


class MultiHeadAttention(nn.Module):
    n_dim: int
    n_head: int
    enc_out: bool

    def setup(self):
        if self.enc_out:
            self.q_proj = nn.Dense(self.n_dim * self.n_head,
                                   kernel_init=nn.initializers.xavier_uniform(),
                                   bias_init=nn.initializers.zeros)

            self.kv_proj = nn.Dense(self.n_dim * self.n_head * 2,
                                    kernel_init=nn.initializers.xavier_uniform(),
                                    bias_init=nn.initializers.zeros)
        else:
            self.qkv_proj = nn.Dense(self.n_dim * self.n_head * 3,
                                     kernel_init=nn.initializers.xavier_uniform(),
                                     bias_init=nn.initializers.zeros)

        self.o_proj = nn.Dense(self.n_dim,
                               kernel_init=nn.initializers.xavier_uniform(),
                               bias_init=nn.initializers.zeros)

        self.layer_norm = nn.LayerNorm()

    def __call__(self, X, enc_out=None, mask=None):
        """
        X: [bs, seq_len, n_dim]
        """
        bs, seq_len, n_dim = X.shape
        if self.enc_out:
            q = self.q_proj(X)
            kv = self.kv_proj(enc_out)
            k, v = jnp.array_split(kv, 2, axis=-1)
        else:
            qkv = self.qkv_proj(X)
            q, k, v = jnp.array_split(qkv, 3, axis=-1)  # [bs, seq_len, n_head, n_dim]

        q = q.reshape((bs, seq_len, self.n_head, n_dim)).transpose(0, 2, 1, 3)  # [bs, n_head, seq_len, n_dim]
        k = k.reshape((bs, seq_len, self.n_head, n_dim)).transpose(0, 2, 1, 3)
        v = v.reshape((bs, seq_len, self.n_head, n_dim)).transpose(0, 2, 1, 3)

        out, attn = scaled_dot_product(q, k, v, mask)
        out = out.swapaxes(1, 2).reshape(bs, seq_len, self.n_head * n_dim)
        out = X + self.o_proj(out)
        out = self.layer_norm(out)
        return out
    

class EncoderLayer(nn.Module):
    conf: dict

    def setup(self):
        self.attn = MultiHeadAttention(self.conf["n_dim"], self.conf["n_head"], False)
        self.lin_norm = LinNorm(self.conf["n_dim"])

    def __call__(self, X):
        out = self.attn(X)
        out = self.lin_norm(out)
        return out
    

class PredLayer(nn.Module):
    conf: dict

    def setup(self):
        self.n_item = self.conf["n_item"]
        self.lin = nn.Dense(self.n_item,
                            kernel_init=nn.initializers.xavier_uniform(),
                            bias_init=nn.initializers.zeros)

    def __call__(self, X, residual_feat):
        out = self.lin(X) + residual_feat
        logits = nn.sigmoid(out)
        return logits
    

class AdaptiveRanking(nn.Module):
    '''
    same architecture with minor differences as CrossCBR and CoHEAT 
    '''
    conf: dict
    ub_graph: sp.coo_matrix
    ui_graph: sp.coo_matrix
    bi_graph: sp.coo_matrix

    def setup(self):
        self.user_emb = self.param('user_emb', 
                                   init_fn=nn.initializers.xavier_uniform(), 
                                   shape=(self.conf["n_user"], self.conf["n_dim"]))
        self.item_emb = self.param('item_emb',
                                   init_fn=nn.initializers.xavier_uniform(), 
                                   shape=(self.conf["n_item"], self.conf["n_dim"]))
        self.bundle_emb = self.param('bundle_emb',
                                     init_fn=nn.initializers.xavier_uniform(), 
                                     shape=(self.conf["n_bundle"], self.conf["n_dim"]))
        self.ub_prop_graph, self.ui_prop_graph, self.bi_prop_graph = self.get_propagate_graph()

    def get_propagate_graph(self):
        # [[uu, ub], [bu, bb]]: [(u+b) x (u+b)]
        ub_propagate_graph = sp.bmat([[sp.coo_matrix((self.ub_graph.shape[0], self.ub_graph.shape[0])), self.ub_graph],
                                      [self.ub_graph.T, sp.coo_matrix((self.ub_graph.shape[1], self.ub_graph.shape[1]))]])
        ub_propagate_graph = sparse.BCOO.from_scipy_sparse(laplace_norm(ub_propagate_graph))
        # [[uu, ui], [iu, ii]]: [(u+i) x (u+i)]
        ui_propagate_graph = sp.bmat([[sp.coo_matrix((self.ui_graph.shape[0], self.ui_graph.shape[0])), self.ui_graph],
                                      [self.ui_graph.T, sp.coo_matrix((self.ui_graph.shape[1], self.ui_graph.shape[1]))]])
        ui_propagate_graph = sparse.BCOO.from_scipy_sparse(laplace_norm(ui_propagate_graph))
        # [bi]: [bxi]
        # print(self.bi_graph.sum())
        bi_propagate_graph = sp.diags(1 / (self.bi_graph.sum(axis=1).A.ravel() + 1e-8)) @ self.bi_graph
        bi_propagate_graph = sparse.BCOO.from_scipy_sparse(bi_propagate_graph)
        # print(ui_propagate_graph.sum(), ub_propagate_graph.sum(), bi_propagate_graph.sum())
        # exit()
        return ub_propagate_graph, ui_propagate_graph, bi_propagate_graph

    def level_propagate(self, feat1, feat2, graph, num_layers=1):
        features = jnp.concatenate([feat1, feat2], axis=0)
        all_features = [features]
        for i in range(0, num_layers):
            features = graph @ features
            features = features / (i+2)
            features = normalize(features)
            all_features.append(features)
        all_features = jnp.stack(all_features, axis=1)
        all_features = jnp.mean(all_features, axis=1)
        feat1, feat2 = jnp.split(all_features, [feat1.shape[0]], axis=0)
        return feat1, feat2

    def propagate(self):
        # propagate bundle level
        u_blevel, b_blevel = self.level_propagate(self.user_emb, self.bundle_emb, self.ub_prop_graph, 1)
        # propagate item level
        u_ilevel, i_ilevel = self.level_propagate(self.user_emb, self.item_emb, self.ui_prop_graph, 1)
        # mean pooling
        b_ilevel = self.bi_prop_graph @ i_ilevel
        u_feats = [u_blevel, u_ilevel]
        b_feats = [b_blevel, b_ilevel]
        return u_feats, b_feats
    
    def cal_c_loss(self, pos, aug, c_temp=0.25):
        pos = normalize(pos)
        aug = normalize(aug)

        pos_score = jnp.sum(pos * aug, axis=1)
        ttl_score = pos @ aug.T

        pos_score = jnp.exp(pos_score / c_temp)
        ttl_score = jnp.sum(jnp.exp(ttl_score) / c_temp, axis=1)
        c_loss = -jnp.mean(jnp.log(pos_score / ttl_score))
        return c_loss
     
    def __call__(self, x):
        u_feats, b_feats = self.propagate()
        '''
        x  [[uid,...], 
            [pbid,...], 
            [nbid,...]]
        '''
        # print(x)
        uid, pbid, nbid = x[0], x[1], x[2]
        u1, u2 = [i[uid] for i in u_feats]
        b1, b2 = [i[pbid] for i in b_feats]
        b3, b4 = [i[nbid] for i in b_feats]

        # u_closs = self.cal_c_loss(u_feats[0], u_feats[1])
        # b_closs = self.cal_c_loss(b_feats[0], b_feats[1])
        # c_loss = (u_closs + b_closs) / 2

        pos_score = jnp.sum(u1 * b1 + u2 * b2, axis=1)
        neg_score = jnp.sum(u1 * b3 + u2 * b4, axis=1)
        loss = -jnp.log(nn.sigmoid(pos_score - neg_score)).mean()
        # return loss + c_loss * 0.04
        return loss
        
    def eval(self, x):
        u_feats, b_feats = self.propagate()
        '''
        x [uid, ...]
        '''
        uid = x
        u1, u2 = [i[uid] for i in u_feats]
        b1, b2 = b_feats
        score = u1 @ b1.T + u2 @ b2.T
        return score


class Net(nn.Module):
    conf: dict

    def setup(self):
        self.n_users = self.conf["n_user"]
        self.n_items = self.conf["n_item"]
        self.n_bundles = self.conf["n_bundle"]
        self.hidden_dim = self.conf["n_dim"]

        self.user_emb = self.param("user_emb", 
                                   nn.initializers.xavier_uniform(),
                                   (self.n_users, self.hidden_dim))

        self.item_emb = self.param("item_emb",
                                   nn.initializers.xavier_uniform(),
                                   (self.n_items, self.hidden_dim))
        
        self.encoder = [EncoderLayer(self.conf) for _ in range(self.conf["n_layer"])]
        self.mlp = PredLayer(self.conf)
        self.enc = nn.Dense(self.hidden_dim,
                            kernel_init=nn.initializers.xavier_uniform(),
                            bias_init=nn.initializers.zeros)

    def __call__(self, uids, prob_iids, prob_iids_bundle):
        """
        uids: user ids
        prob_iids: user's item probability
        prob_iids_bundle: sampled item in interacted bundle probability (noise while inference)
        """
        # print(uids)
        users_feat = self.user_emb[uids]
        prob_enc = self.enc(prob_iids_bundle)
        in_feat = jnp.concat([users_feat, prob_enc], axis=1)
        out_feat = self.mlp(in_feat, prob_iids)
        return out_feat
        # return prob_iids
    
