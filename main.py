from argparse import ArgumentParser

import jax
import optax
from flax import linen as nn
from flax.training import train_state
from tqdm import tqdm

from model import Net
from utils import *
from utils import DiffusionScheduler

TOTAL_TIMESTEP = conf["timestep"]
INF = 1e8


def get_args():
    argp = ArgumentParser()
    argp.add_argument("--device_id", type=int, default=0)
    argp.add_argument("--dataset", type=str, default="clothing")
    argp.add_argument("--data_path", type=str, default="datasets/cold_bundles")
    args = argp.parse_args()
    return args


def cal_metrics(
        all_gen_buns_batch,
        ub_mask_graph_batch,
        ub_mat, bi_mat,
        topk
):
    recall_cnt, pre_cnt, ndcg_cnt, cnt = 0, 0, 0, 0
    pred_score = all_gen_buns_batch @ bi_mat.T
    ub_mask_graph_batch = ub_mask_graph_batch.todense()

    score = pred_score + ub_mask_graph_batch * -INF
    bs = score.shape[0]
    _, col_ids = jax.lax.top_k(score, k=topk)
    row_ids = jnp.broadcast_to(jnp.arange(0, bs).reshape(-1, 1), (bs, topk))
    hit = ub_mat[row_ids, col_ids].todense()

    # recall
    recall_cnt = hit.sum(axis=1) / (ub_mat.sum(axis=1) + 1e-8)

    # precision
    pre_cnt = hit.sum(axis=1) / topk

    # ndcg
    def DCG(hit, topk):
        dcg = hit / jnp.broadcast_to(jnp.log2(jnp.arange(2, topk + 2)), hit.shape)
        return dcg.sum(axis=-1)

    def IDCG(num_pos, topk):
        temp_hit = np.zeros(topk)
        temp_hit[:num_pos] = 1
        return DCG(temp_hit, topk)

    IDCGs = [0] * (topk + 1)
    IDCGs[0] = 1
    for i in range(1, topk + 1):
        IDCGs[i] = IDCG(i, topk)

    IDCGs = np.array(IDCGs)
    num_pos_clamp = jax.lax.clamp(0, ub_mat.sum(axis=1).astype(jnp.int32), topk).astype(jnp.int32)
    dcg = DCG(hit, topk).reshape(-1, 1)
    idcg = IDCGs[num_pos_clamp]
    ndcg_cnt = dcg / idcg

    return recall_cnt.sum(), pre_cnt.sum(), ndcg_cnt.sum()


def kl_divergence(src, trg):
    kl_div = trg * (jnp.log(trg) - jnp.log(src))
    return jnp.sum(kl_div)


def mse(x, y):
    return jnp.mean((x - y) ** 2)


def train_step(state, uids, prob_iids, noisy_prob_iids_bundle, prob_iids_bundle):
    def loss_fn(params, uids, prob_iids, noisy_prob_iids_bundle, prob_iids_bundle):
        logits = state.apply_fn(params, uids, prob_iids, noisy_prob_iids_bundle)
        mse_loss = mse(logits, prob_iids_bundle)  # MSE

        slogits = nn.softmax(logits)
        sprob_iids = nn.softmax(prob_iids)
        kl_loss = kl_divergence(slogits, sprob_iids)  # Kullback-Leibler Divergence (true probability: prob_iids)

        loss = mse_loss + kl_loss
        return loss, {"loss": loss, "mse": mse_loss, "kl": kl_loss}

    aux, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params, uids, prob_iids, noisy_prob_iids_bundle,
                                                           prob_iids_bundle)
    state = state.apply_gradients(grads=grads)
    loss, aux_dict = aux
    return state, loss, aux_dict


def train(state, dataloader, noise_scheduler, epochs, device, key):
    print("TRAINING")

    for epoch in range(epochs):
        pbar = tqdm(dataloader)
        for uids, prob_iids, prob_iids_bundle in pbar:
            uids = jnp.array(uids, dtype=jnp.int32)
            prob_iids = jnp.array(prob_iids, dtype=jnp.float32)
            prob_iids_bundle = jnp.array(prob_iids_bundle, jnp.float32)

            randkey, timekey, key = jax.random.split(key, num=3)
            noise = jax.random.normal(randkey, shape=prob_iids_bundle.shape)
            timestep = jax.random.randint(timekey, (prob_iids_bundle.shape[0],), minval=0, maxval=TOTAL_TIMESTEP - 1)

            noisy_prob_iids_bundle = noise_scheduler.add_noise(prob_iids_bundle, noise, timestep)
            state, loss, aux_dict = jax.jit(train_step, device=device)(state, uids, prob_iids, noisy_prob_iids_bundle,
                                                                       prob_iids_bundle)
            pbar.set_description("EPOCH: %i | LOSS: %.4f | KL_LOSS: %.4f | MSE_LOSS: %.4f" % (
                epoch, aux_dict["loss"], aux_dict["kl"], aux_dict["mse"]))
    return state


def inference(model, state, test_dataloader, noise_scheduler, key, n_item):
    all_genbundles = []
    for test_data in test_dataloader:
        key, rand_key = jax.random.split(key)
        uids, prob_iids = test_data
        uids = jnp.array(uids, dtype=jnp.int32)
        prob_iids = jnp.array(prob_iids, jnp.float32)
        noisy_prob_iids_bundle = jax.random.normal(rand_key, shape=(uids.shape[0], n_item))

        post_prob_iids_bundle = noisy_prob_iids_bundle
        for i, t in enumerate(noise_scheduler.timestep):
            model_output = model.apply(state.params, uids, prob_iids, post_prob_iids_bundle)
            post_prob_iids_bundle = noise_scheduler.step(model_output, t, post_prob_iids_bundle)

        all_genbundles.append(post_prob_iids_bundle)
    all_genbundles = np.concatenate(all_genbundles, axis=0)
    return all_genbundles


def eval(conf, test_data, all_gen_buns):
    batch_size = conf["batch_size"]
    ui_mat = test_data.ui_graph
    bi_mat = test_data.bi_graph
    ub_mask_graph = test_data.ub_graph_train
    ub_mat = test_data.ub_graph_test

    uids_test = test_data.test_uid
    batch_idx = np.arange(0, len(uids_test))
    test_batch_loader = DataLoader(batch_idx, batch_size=batch_size, shuffle=False, drop_last=False)

    for topk in [10, 20, 40, 50]:
        recall_cnt = 0
        pre_cnt = 0
        ndcg_cnt = 0

        for batch in test_batch_loader:
            start = batch[0]
            end = batch[-1]

            uids_test_batch = uids_test[start:end + 1]
            ub_mask_graph_batch = ub_mask_graph[uids_test_batch]
            # all_gen_buns_batch = all_gen_buns[uids_test_batch]
            all_gen_buns_batch = all_gen_buns[start:end + 1]

            r_cnt, p_cnt, n_cnt = cal_metrics(all_gen_buns_batch,
                                              ub_mask_graph_batch,
                                              ub_mat[uids_test_batch],
                                              bi_mat,
                                              topk)
            recall_cnt += r_cnt
            pre_cnt += p_cnt
            ndcg_cnt += n_cnt

        res_line = "Recall@%i: %.4f" % (topk, recall_cnt / len(uids_test)) + \
                   " | " + "Precision@%i: %.4f" % (topk, pre_cnt / len(uids_test)) + \
                   " | " + "NDCG@%i: %.4f" % (topk, ndcg_cnt / len(uids_test))
        print(res_line)


def main():
    """
    Load Config & Init
    """
    args = get_args()
    dataset_name = args.dataset
    conf["dataset"] = args.dataset
    conf["data_path"] = args.data_path
    nu, nb, ni = get_size(f"{conf['data_path']}/{dataset_name}/{dataset_name}_data_size.txt")
    conf["n_user"] = nu
    conf["n_item"] = ni
    conf["n_bundle"] = nb
    devices = jax.devices()
    device = devices[args.device_id]
    conf["device"] = device

    rng_infer, rng_gen, rng_model = jax.random.split(jax.random.PRNGKey(2025), num=3)
    np.random.seed(2025)
    print(conf)

    """
    Construct Training/Validating/Testing Data
    """
    train_data = TrainDataVer2(conf)
    test_data = TestData(conf, "test")
    valid_data = TestData(conf, "tune")
    """
    Main Model & Optimizer, Train State
    """
    sample_uids = jnp.array([0])
    sample_prob_iids = jnp.empty((1, conf["n_item"]))
    sample_prob_iids_bundle = jnp.empty((1, conf["n_item"]))
    model = Net(conf, train_data.ui_graph)

    conf["model_name"] = model.__class__.__name__
    print(f"MODEL NAME: {conf['model_name']}")
    print(f"DATACLASS: {train_data.__class__.__name__}, {test_data.__class__.__name__}({test_data.task})")
    params = model.init(rng_model, sample_uids, sample_prob_iids, sample_prob_iids_bundle)
    param_count = sum(x.size for x in jax.tree.leaves(params))
    print("#PARAMETERS:", param_count)
    optimizer = optax.adam(learning_rate=1e-3)

    state = train_state.TrainState.create(apply_fn=model.apply,
                                          params=params,
                                          tx=optimizer)
    noise_scheduler = DiffusionScheduler(num_train_timestep=TOTAL_TIMESTEP)

    dataloader = DataLoader(train_data,
                            batch_size=conf["batch_size"],
                            shuffle=True,
                            drop_last=True)

    test_dataloader = DataLoader(test_data,
                                 batch_size=conf["batch_size"],
                                 shuffle=False,
                                 drop_last=False)

    valid_dataloader = DataLoader(valid_data,
                                  batch_size=conf["batch_size"],
                                  shuffle=False,
                                  drop_last=False)

    """
    Training & Save checkpoint
    """
    state = train(state, dataloader, noise_scheduler, conf["epoch"], device, rng_gen)
    """
    Generate & Evaluate
    """
    rng_infer_test, rng_infer_valid = jax.random.split(rng_infer)

    print("VALIDATING")
    generated_bundles_valid = inference(model, state, valid_dataloader, noise_scheduler, rng_infer_valid,
                                        conf["n_item"])
    eval(conf, valid_data, generated_bundles_valid)

    print("TESTING")
    generated_bundles_test = inference(model, state, test_dataloader, noise_scheduler, rng_infer_test, conf["n_item"])
    eval(conf, test_data, generated_bundles_test)


if __name__ == "__main__":
    main()
