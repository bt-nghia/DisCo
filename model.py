from flax import linen as nn

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

    def __call__(self, X):
        return X
    