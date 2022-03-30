import torch
import utils
from torch.utils.data import DataLoader
from data_helper import BiDataloader, EvalDataset


def train_step(model, data, kg, optimizer, scheduler):
    """
    A single train step. Apply back-propation and return the loss.
    :param model:
    :param data:
    :param kg:
    :param optimizer:
    :param scheduler:
    :return:
    """

    cfg = utils.get_global_config()
    device = torch.device(cfg.device)
    model.train()
    optimizer.zero_grad()

    (src, rel, _), label, rm_edges = data
    src, rel, label, rm_edges = src.to(device), rel.to(device), label.to(device), rm_edges.to(device)
    # randomly remove the training edges
    if cfg.rm_rate > 0:
        kg.remove_edges(rm_edges)
    score = model(src, rel, kg)
    loss = model.loss(score, label)
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()

    log = {
        'loss': loss.item()
    }

    return log


def evaluate(model, set_flag, kg, record=False) -> dict:
    """
    Evaluate the dataset.
    :param model: model to be evaluated.
    :param set_flag: train / valid / test set data to be evaluated.
    :param kg: kg used to aggregate the embedding.
    :param record: whether to record the rank of all the data.
    :return:
    """
    assert set_flag in ['train', 'valid', 'test']
    model.eval()
    cfg = utils.get_global_config()
    dataset = cfg.dataset
    n_ent = utils.DATASET_STATISTICS[dataset]['n_ent']
    device = torch.device(cfg.device)

    eval_h_loader = DataLoader(
        dataset=EvalDataset(set_flag, 'tail_batch'),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.cpu_worker_num,
        collate_fn=EvalDataset.collate_fn
    )
    eval_t_loader = DataLoader(
        EvalDataset(set_flag, 'head_batch'),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.cpu_worker_num,
        collate_fn=EvalDataset.collate_fn
    )
    eval_loader = BiDataloader(eval_h_loader, eval_t_loader)

    hb_metrics, tb_metrics, avg_metrics = {}, {}, {}
    metrics = {
        'head_batch': hb_metrics,
        'tail_batch': tb_metrics,
        'average': avg_metrics,
        'ranking': []
    }
    hits_range = [1, 3, 10, 100, 1000, round(0.5*n_ent)]
    with torch.no_grad():
        # aggregate the embedding
        ent_emb, rel_emb = model.aggragate_emb(kg)
        for i, data in enumerate(eval_loader):
            # filter_bias: (bs, n_ent)
            (src, rel, dst), filter_bias, mode = data
            src, rel, dst, filter_bias = src.to(device), rel.to(device), dst.to(device), filter_bias.to(device)
            # (bs, n_ent)
            score = model.predictor(ent_emb[src], rel_emb[rel], ent_emb)
            score = score + filter_bias

            pos_inds = dst
            batch_size = filter_bias.shape[0]
            pos_score = score[torch.arange(batch_size), pos_inds].unsqueeze(dim=1)
            # compare the positive value with negative values to compute rank
            # when values equal, take the mean of upper and lower bound
            compare_up = torch.gt(score, pos_score)  # (bs, entity_num), >
            compare_low = torch.ge(score, pos_score)  # (bs, entity_num), >=
            ranking_up = compare_up.to(dtype=torch.float).sum(dim=1) + 1  # (bs, )
            ranking_low = compare_low.to(dtype=torch.float).sum(dim=1)  # include the pos one itself, no need to +1
            ranking = (ranking_up + ranking_low) / 2
            if record:
                rank = torch.stack([src, rel, dst, ranking], dim=1)  # (bs, 4)
                metrics['ranking'].append(rank)

            results = metrics[mode]
            results['MR'] = results.get('MR', 0.) + ranking.sum().item()
            results['MRR'] = results.get('MRR', 0.) + (1 / ranking).sum().item()
            for k in hits_range:
                results['HITS@{}'.format(k)] = results.get('HITS@{}'.format(k), 0.) + \
                                               (ranking <= k).to(torch.float).sum().item()
            results['n_data'] = results.get('n_data', 0) + batch_size

        assert metrics['head_batch']['n_data'] == metrics['tail_batch']['n_data']

        for k, results in metrics.items():
            if k in ['ranking', 'average']:
                continue
            results['MR'] /= results['n_data']
            results['MRR'] /= results['n_data']
            for j in hits_range:
                results['HITS@{}'.format(j)] /= results['n_data']

        # average the hb and tb values to get the final reports
        for k, value in metrics['head_batch'].items():
            metrics['average'][k] = (metrics['head_batch'][k] + metrics['tail_batch'][k]) / 2

        # sort in the ranking order
        if record:
            metrics['ranking'] = torch.cat(metrics['ranking'], dim=0).tolist()
            metrics['ranking'] = sorted(metrics['ranking'], key=lambda x: x[3], reverse=True)

    return metrics

