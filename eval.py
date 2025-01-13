import jax.lax
import numpy as np
import jax
import jax.numpy as jnp
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils import get_pairs, list2csr_sp_graph

import time
import pickle

INF = 1e8


def eval(all_gen_buns_batch, 
         ub_mask_graph_batch, 
         ub_mat, bi_mat,
         topk):
    
    recall_cnt, pre_cnt, ndcg_cnt, cnt = 0, 0, 0, 0
    pred_score = all_gen_buns_batch @ bi_mat.T
    ub_mask_graph_batch = ub_mask_graph_batch.todense()

    score = pred_score + ub_mask_graph_batch * -INF
    bs = score.shape[0]
    _, col_ids = jax.lax.top_k(score, k=topk)
    row_ids = jnp.broadcast_to(jnp.arange(0, bs).reshape(-1, 1), (bs, topk))
    hit = ub_mat[row_ids, col_ids].todense()

    #recall
    recall_cnt = hit.sum(axis=1) / (ub_mat.sum(axis=1) + 1e-8)
    
    #precision
    pre_cnt = hit.sum(axis=1) / topk
    
    #ndcg
    def DCG(hit, topk):
        dcg = hit / jnp.broadcast_to(jnp.log2(jnp.arange(2, topk+2)), hit.shape)
        return dcg.sum(axis=-1)

    def IDCG(num_pos, topk):
        temp_hit = np.zeros(topk)
        temp_hit[:num_pos] = 1
        return DCG(temp_hit, topk)
    
    IDCGs = [0] * (topk+1)
    IDCGs[0] = 1
    for i in range(1, topk+1):
        IDCGs[i] = IDCG(i, topk)

    IDCGs = np.array(IDCGs)
    num_pos_clamp = jax.lax.clamp(0, ub_mat.sum(axis=1).astype(jnp.int32), topk).astype(jnp.int32)
    dcg = DCG(hit, topk).reshape(-1, 1)
    idcg = IDCGs[num_pos_clamp]
    ndcg_cnt = dcg / idcg

    return recall_cnt.sum(), pre_cnt.sum(), ndcg_cnt.sum()


def cal_jaccard(mat1, mat2):
    intersect = mat1 @ mat2.T
    total1 = mat1.sum(axis=1)
    total2 = mat2.sum(axis=1)
    total = total1.reshape(-1, 1) + total2.reshape(1, -1) - intersect
    jaccard_score = intersect / (total + 1e-8)
    return jaccard_score


def cal_cosine(mat1, mat2):
    m, n = mat1.shape[0], mat2.shape[0]
    nume = mat1 @ mat2.T
    norm1 = jnp.linalg.norm(mat1, axis=1).reshape(-1, 1) #[m ,1]
    norm2 = jnp.linalg.norm(mat2, axis=1).reshape(1, -1) #[1, n]
    norm1 = jnp.broadcast_to(norm1, (m, n))
    norm2 = jnp.broadcast_to(norm2, (m, n))
    denom =  norm1 * norm2 + 1e-8
    cosine = nume / denom
    return cosine


if __name__ == "__main__":

    argp = ArgumentParser()
    argp.add_argument("--model", type=str, default="BRIDGE")
    argp.add_argument("--dataset", type=str, default="clothing")
    argp.add_argument("--batch_size", type=int, default=2048)
    argp.add_argument("--task", type=str, default="test")
    args = argp.parse_args()

    dataset = args.dataset
    task = args.task
    batch_size = args.batch_size
    model_name = args.model

    # lazy-load
    if dataset == "clothing":
        nu, nb, ni = 965, 1910, 4487
    elif dataset == "electronic":
        nu, nb, ni = 888, 1750, 3499
    elif dataset == "food":
        nu, nb, ni = 879, 1784, 3767
    elif dataset == "Steam":
        nu, nb, ni = 29634, 615, 2819
    elif dataset == "iFashion":
        nu, nb, ni = 53897, 27694, 42563
    elif dataset == "NetEase":
        nu, nb, ni = 18528, 22864, 123628
    elif dataset == "meal":
        nu, nb, ni = 1575, 3817, 7280
    elif dataset == "Youshu":
        nu, nb, ni = 8039, 4771, 32770
    else:
        raise ValueError("Invalid datasets")

    bi_pairs = get_pairs(f"datasets/{dataset}/bundle_item.txt")
    bi_mat = list2csr_sp_graph(bi_pairs, (nb, ni))

    ui_pairs = get_pairs(f"datasets/{dataset}/user_item.txt")
    ui_mat = list2csr_sp_graph(ui_pairs, (nu, ni))

    ub_mask_pair = get_pairs(f"datasets/{dataset}/user_bundle_train.txt")
    ub_mask_graph = list2csr_sp_graph(ub_mask_pair, (nu, nb))

    ub_pairs = get_pairs(f"datasets/{dataset}/user_bundle_{task}.txt")
    ub_mat = list2csr_sp_graph(ub_pairs, (nu, nb)) # all dense graph

    all_gen_buns = ui_mat
    # with open(f"datasets/{dataset}/generated_bundles_list_{task}.pkl", "rb") as f: # generated bundles for user in test 
        # all_gen_buns = pickle.load(f)

    uids_test = ub_pairs[:, 0]
    uids_test = np.unique(uids_test)
    uids_test.sort()
    num_batch = int(len(uids_test) / batch_size)
    batch_idx = np.arange(0, len(uids_test))
    test_batch_loader = DataLoader(batch_idx, batch_size=batch_size, shuffle=False, drop_last=False)

    start_time = time.time()

    for topk in [1, 2, 3, 5, 10, 20, 40, 50]:
        recall_cnt = 0
        pre_cnt = 0
        ndcg_cnt = 0

        for batch in test_batch_loader:
            start = batch[0]
            end = batch[-1]

            uids_test_batch = uids_test[start:end+1]
            ub_mask_graph_batch = ub_mask_graph[uids_test_batch]
            all_gen_buns_batch = all_gen_buns[uids_test_batch]
            # all_gen_buns_batch = all_gen_buns[start:end+1]
            
            r_cnt, p_cnt, n_cnt = eval(all_gen_buns_batch,
                                    ub_mask_graph_batch, ub_mat[uids_test_batch], bi_mat, 
                                    topk)
            recall_cnt+=r_cnt
            pre_cnt+=p_cnt
            ndcg_cnt+=n_cnt

        print("Recall@%i: %s" %(topk, recall_cnt / len(uids_test)))
        print("Precision@%i: %s" %(topk, pre_cnt / len(uids_test)))
        print("NDCG@%i: %s" %(topk, ndcg_cnt / len(uids_test)))
    print("EVALUATE TIME: %s seconds" % (time.time() - start_time))
    