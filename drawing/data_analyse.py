import sys
# add module search path
sys.path.append('code')

import hydra
from collections import defaultdict
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, exists
from omegaconf import DictConfig
import utils
from data_helper import read_data, construct_kg, get_kg


class Metrics:
    def __init__(self):
        self.cfg = utils.get_global_config()
        device = self.cfg.device
        src, dst, rel, _, _ = construct_kg('train', directed=True)
        self.kg = get_kg(src, dst, rel, device)

    def compute_semantic_metrics(self, save_path):
        """
        compute three semantic evidence metrics
        M_rt: relation SE metric
        M_ht: entity SE metric
        M_hrt: triple SE metric
        :return:
        """
        # data preparation
        train_d = read_data('train')
        test_d = read_data('test')
        # comupte similary between entites
        ent2sim = Metrics.compute_ent_simlarity()
        # data to semantic evidence dict
        # key: (h, r, t, mode), value: {'M_rt': xx, 'M_ht': xx, 'M_hrt': xx}
        data2metrics = defaultdict(dict)
        for (h, r), t_set in test_d['pos_tails'].items():
            for t in t_set:
                # M_rt: relation SE metric, co-occurence between r & t in train set
                mrt = len(train_d['pos_heads'].get((r, t), []))
                # M_ht: entity SE metric, path num from h to t in train set
                mht = Metrics.count_ent_path(self.kg, h, t)
                # M_hrt: similarity between t and other answer entities in train set
                mhrt = 0
                answer_set = train_d['pos_tails'].get((h, r), set())
                for e in answer_set:
                    e1, e2 = min(t, e), max(t, e)
                    mhrt += ent2sim.get((e1, e2), 0)
                data2metrics[(h, r, t, 'tail-batch')]['M_rt'] = mrt
                data2metrics[(h, r, t, 'tail-batch')]['M_ht'] = mht
                data2metrics[(h, r, t, 'tail-batch')]['M_hrt'] = mhrt
        for (r, t), h_set in test_d['pos_heads'].items():
            for h in h_set:
                # M_rt
                mrt = len(train_d['pos_tails'].get((h, r), []))
                # M_ht
                mht = Metrics.count_ent_path(self.kg, t, h)
                # M_hrt
                mhrt = 0
                answer_set = train_d['pos_heads'].get((r, t), set())
                for e in answer_set:
                    e1, e2 = min(h, e), max(h, e)
                    mhrt += ent2sim.get((e1, e2), 0)
                data2metrics[(h, r, t, 'head-batch')]['M_rt'] = mrt
                data2metrics[(h, r, t, 'head-batch')]['M_ht'] = mht
                data2metrics[(h, r, t, 'head-batch')]['M_hrt'] = mhrt
        pickle.dump(data2metrics, open(save_path, 'wb'))

    def plot_se2rank_6bar(self, model_name: list, rank_load_paths: list, fig_save_path, se_metric_path):
        assert len(model_name) == 6

        # each model's pred rank
        data2ranks = []
        for p in rank_load_paths:
            # key: (h, r, t, mode), value: rank
            data2ranks.append(pickle.load(open(p, 'rb')))

        # key: (h, r, t, mode), value: {'M_rt': xx, 'M_ht': xx, 'M_hrt': xx}
        data2se = pickle.load(open(se_metric_path, 'rb'))
        for d2r in data2ranks:
            assert len(d2r) == len(data2se)
        total_ndata = len(data2se)

        # get the se metric and model pred rank, save as the list respectively
        mrt_list, mht_list, mhrt_list, rank_list = [], [], [], [[] for _ in data2ranks]
        for d, metrics in data2se.items():
            mrt_list.append(metrics['M_rt'])
            mht_list.append(metrics['M_ht'])
            mhrt_list.append(metrics['M_hrt'])
            for i, d2r in enumerate(data2ranks):
                # get the corresponding pred rank of the triple
                assert d in d2r, d
                rank_list[i].append(d2r[d])
        for i, rank in enumerate(rank_list):
            rank_list[i] = np.array(rank)

        # split the data into SE metric ranges
        if self.cfg.dataset == 'FB15k_237':
            # range bound
            mrt_bounds = [0, 3, 40]
            mht_bounds = [0, 1, 3]
            mhrt_bounds = [0, 4, 50]
        elif self.cfg.dataset == 'WN18RR':
            mrt_bounds = [0, 1]
            mht_bounds = [0, 1]
            mhrt_bounds = [0, 1]
        else:
            raise NotImplementedError
        mrt_range2ndata, mrt_range2id = Metrics.split_data(mrt_list, mrt_bounds)
        mht_range2ndata, mht_range2id = Metrics.split_data(mht_list, mht_bounds)
        mhrt_range2ndata, mhrt_range2id = Metrics.split_data(mhrt_list, mhrt_bounds)
        # print('M_rt range statistics: {}'.format(mrt_range2ndata))
        # print('M_ht range statistics: {}'.format(mht_range2ndata))
        # print('M_hrt range statistics: {}'.format(mhrt_range2ndata))

        # compute the proportion of each SE range
        mrt_ranges, mrt_ndata = [], []
        for r_str, r_ndata in mrt_range2ndata.items():
            mrt_ranges.append(r_str)
            mrt_ndata.append('#:{:.1%}'.format(r_ndata / total_ndata))

        mht_ranges, mht_ndata = [], []
        for r_str, r_ndata in mht_range2ndata.items():
            mht_ranges.append(r_str)
            mht_ndata.append('#:{:.1%}'.format(r_ndata / total_ndata))

        mhrt_ranges, mhrt_ndata = [], []
        for r_str, r_ndata in mhrt_range2ndata.items():
            mhrt_ranges.append(r_str)
            mhrt_ndata.append('#:{:.1%}'.format(r_ndata / total_ndata))

        # compute the Mean Rank in each SE range
        mrt_ranks, mht_ranks, mhrt_ranks = [], [], []
        for rank in rank_list:
            mrt_r, mht_r, mhrt_r = [], [], []
            for _, r_ids in mrt_range2id.items():
                mrt_r.append(np.mean(rank[r_ids]))
            for _, r_ids in mht_range2id.items():
                mht_r.append(np.mean(rank[r_ids]))
            for _, r_ids in mhrt_range2id.items():
                mhrt_r.append(np.mean(rank[r_ids]))
            mrt_ranks.append(mrt_r)
            mht_ranks.append(mht_r)
            mhrt_ranks.append(mhrt_r)

        # Plot
        plt.figure(figsize=(15, 5))
        n_model = len(model_name)
        if self.cfg.dataset == 'FB15k_237':
            b_width = 0.6
            inds = np.array([4, 8, 12])
            zeros = np.array([0, 0, 0])
        elif self.cfg.dataset == 'WN18RR':
            b_width = 0.5
            inds = np.array([4, 8])
            zeros = np.array([0, 0])
        else:
            raise NotImplementedError
        offset_inds = inds + (n_model - 1) / 2 * b_width

        ax1 = plt.subplot(131)
        plt.xlabel('Relation level SE Ranges', size='x-large')
        plt.ylabel('Mean Rank', size='x-large')
        for i, (rank, name) in enumerate(zip(mrt_ranks, model_name)):
            plt.bar(inds+i*b_width, rank, width=b_width, label=name)
            for (x, y) in zip(inds + i * b_width, rank):
                if y < 100 and self.cfg.dataset == 'WN18RR':
                    plt.text(x, y + 0.05, round(y, None), size='small', ha='center', va='bottom')
        plt.xticks(offset_inds, mrt_ranges, size='large', rotation=0)
        plt.tick_params(axis='y', labelsize='large')
        plt.legend(fontsize=10)
        ax1.twiny()
        plt.bar(offset_inds, zeros)
        plt.xticks(offset_inds, mrt_ndata, size='large', rotation=0)

        ax2 = plt.subplot(132, sharey=ax1)
        plt.xlabel('Entity level SE Ranges', size='x-large')
        for i, (rank, name) in enumerate(zip(mht_ranks, model_name)):
            plt.bar(inds + i * b_width, rank, width=b_width, label=name)
            for (x, y) in zip(inds + i * b_width, rank):
                if y < 100 and self.cfg.dataset == 'WN18RR':
                    plt.text(x, y + 0.05, round(y, None), size='small', ha='center', va='bottom')
        plt.xticks(offset_inds, mht_ranges, size='large', rotation=0)
        plt.tick_params(axis='y', labelsize='large')
        plt.legend(fontsize=10)
        ax2.twiny()
        plt.bar(offset_inds, zeros)
        plt.xticks(offset_inds, mht_ndata, size='large', rotation=0)

        ax3 = plt.subplot(133, sharey=ax1)
        plt.xlabel('Triple level SE Ranges', size='x-large')
        for i, (rank, name) in enumerate(zip(mhrt_ranks, model_name)):
            plt.bar(inds + i * b_width, rank, width=b_width, label=name)
            for (x, y) in zip(inds + i * b_width, rank):
                if y < 100 and self.cfg.dataset == 'WN18RR':
                    plt.text(x, y + 0.05, round(y, None), size='small', ha='center', va='bottom')
        plt.xticks(offset_inds, mhrt_ranges, size='large', rotation=0)
        plt.tick_params(axis='y', labelsize='large')
        plt.legend(fontsize=10)
        ax3.twiny()
        plt.bar(offset_inds, zeros)
        plt.xticks(offset_inds, mhrt_ndata, size='large', rotation=0)

        plt.subplots_adjust(wspace=0.2)  # distance of sub-figs
        plt.savefig(fig_save_path, format='svg')

    def plot_se2rank_2bar(self, model_name: list, rank_load_paths: list, fig_save_path, se_metric_path):
        assert len(model_name) == 2

        # each model's pred rank
        data2ranks = []
        for p in rank_load_paths:
            # key: (h, r, t, mode), value: rank
            data2ranks.append(pickle.load(open(p, 'rb')))

        # key: (h, r, t, mode), value: {'M_rt': xx, 'M_ht': xx, 'M_hrt': xx}
        data2se = pickle.load(open(se_metric_path, 'rb'))
        for d2r in data2ranks:
            assert len(d2r) == len(data2se)
        total_ndata = len(data2se)

        # get the se metric and model pred rank, save as the list respectively
        mrt_list, mht_list, mhrt_list, rank_list = [], [], [], [[] for _ in data2ranks]
        for d, metrics in data2se.items():
            mrt_list.append(metrics['M_rt'])
            mht_list.append(metrics['M_ht'])
            mhrt_list.append(metrics['M_hrt'])
            for i, d2r in enumerate(data2ranks):
                # get the corresponding pred rank of the triple
                assert d in d2r, d
                rank_list[i].append(d2r[d])
        for i, rank in enumerate(rank_list):
            rank_list[i] = np.array(rank)

        # split the data into SE metric ranges
        # range bound
        mrt_bounds = [0, 3, 40]
        mht_bounds = [0, 1, 3]
        mhrt_bounds = [0, 4, 50]
        mrt_range2ndata, mrt_range2id = Metrics.split_data(mrt_list, mrt_bounds)
        mht_range2ndata, mht_range2id = Metrics.split_data(mht_list, mht_bounds)
        mhrt_range2ndata, mhrt_range2id = Metrics.split_data(mhrt_list, mhrt_bounds)

        # compute the proportion of each SE range
        mrt_ranges, mrt_ndata = [], []
        for r_str, r_ndata in mrt_range2ndata.items():
            mrt_ranges.append(r_str)
            mrt_ndata.append('#:{:.1%}'.format(r_ndata / total_ndata))

        mht_ranges, mht_ndata = [], []
        for r_str, r_ndata in mht_range2ndata.items():
            mht_ranges.append(r_str)
            mht_ndata.append('#:{:.1%}'.format(r_ndata / total_ndata))

        mhrt_ranges, mhrt_ndata = [], []
        for r_str, r_ndata in mhrt_range2ndata.items():
            mhrt_ranges.append(r_str)
            mhrt_ndata.append('#:{:.1%}'.format(r_ndata / total_ndata))

        # compute the Mean Rank in each SE range
        mrt_ranks, mht_ranks, mhrt_ranks = [], [], []
        for rank in rank_list:
            mrt_r, mht_r, mhrt_r = [], [], []
            for _, r_ids in mrt_range2id.items():
                mrt_r.append(np.mean(rank[r_ids]))
            for _, r_ids in mht_range2id.items():
                mht_r.append(np.mean(rank[r_ids]))
            for _, r_ids in mhrt_range2id.items():
                mhrt_r.append(np.mean(rank[r_ids]))
            mrt_ranks.append(mrt_r)
            mht_ranks.append(mht_r)
            mhrt_ranks.append(mhrt_r)

        # Plot
        # two bars
        plt.figure(figsize=(10, 5))
        n_model = len(model_name)
        assert self.cfg.dataset == 'FB15k_237'
        b_width = 0.4
        inds = np.array([1, 2, 3])
        zeros = np.array([0, 0, 0])
        offset_inds = inds + (n_model - 1) / 2 * b_width

        ax1 = plt.subplot(131)
        plt.xlabel('Relation level SE Ranges', size='x-large')
        plt.ylabel('Mean Rank', size='xx-large')
        for i, (rank, name) in enumerate(zip(mrt_ranks, model_name)):
            plt.bar(inds+i*b_width, rank, width=b_width, label=name)
            # for (x, y) in zip(inds + i * b_width, rank):
            #     plt.text(x, y + 0.05, round(y, n_round), size='xx-small', ha='center', va='bottom')
        plt.xticks(offset_inds, mrt_ranges, size='large', rotation=0)
        plt.tick_params(axis='y', labelsize='large')
        plt.legend(fontsize=12)
        # ax1.twiny()
        # plt.bar(offset_inds, zeros)
        # plt.xticks(offset_inds, mrt_ndata, size='large', rotation=0)

        ax2 = plt.subplot(132, sharey=ax1)
        plt.xlabel('Entity level SE Ranges', size='x-large')
        for i, (rank, name) in enumerate(zip(mht_ranks, model_name)):
            plt.bar(inds + i * b_width, rank, width=b_width, label=name)
            # for (x, y) in zip(inds + i * b_width, rank):
            #     plt.text(x, y + 0.05, round(y, n_round), size='xx-small', ha='center', va='bottom')
        plt.xticks(offset_inds, mht_ranges, size='large', rotation=0)
        plt.tick_params(axis='y', labelsize='large')
        plt.legend(fontsize=12)
        # ax2.twiny()
        # plt.bar(offset_inds, zeros)
        # plt.xticks(offset_inds, mht_ndata, size='large', rotation=0)

        ax3 = plt.subplot(133, sharey=ax1)
        plt.xlabel('Triple level SE Ranges', size='x-large')
        for i, (rank, name) in enumerate(zip(mhrt_ranks, model_name)):
            plt.bar(inds + i * b_width, rank, width=b_width, label=name)
            # for (x, y) in zip(inds + i * b_width, rank):
            #     plt.text(x, y + 0.05, round(y, n_round), size='xx-small', ha='center', va='bottom')
        plt.xticks(offset_inds, mhrt_ranges, size='large', rotation=0)
        plt.tick_params(axis='y', labelsize='large')
        plt.legend(fontsize=12)
        # ax3.twiny()
        # plt.bar(offset_inds, zeros)
        # plt.xticks(offset_inds, mhrt_ndata, size='large', rotation=0)

        plt.subplots_adjust(wspace=0.25)  # distance of sub-figs
        plt.savefig(fig_save_path, format='svg')

    @staticmethod
    def compute_ent_simlarity(set_flag='train'):
        """
        compute similarity between entites based on the common neighbors
        :param set_flag: train / valid / test set flag
        :return:
        """
        d = read_data(set_flag)
        pos_tails, pos_heads = d['pos_tails'], d['pos_heads']
        pair2w = dict()
        # to control the complexity, we don't conisder the neighbors with large connections,
        # like the high frewquency context words 'a', 'an', 'the' in natural language,
        # which provide little measurement inforamtion
        max_size = 25
        # O(n^2)
        # common head neighbors
        for (h, r), tails in pos_tails.items():
            tails, n = list(tails), len(tails)
            if n > max_size:
                continue
            for i in range(n):
                for j in range(i + 1, n):
                    # set the place order within the entity pair
                    t1, t2 = min(tails[i], tails[j]), max(tails[i], tails[j])
                    pair2w[(t1, t2)] = pair2w.setdefault((t1, t2), 0) + 1

        # common tail neighbors
        for (r, t), heads in pos_heads.items():
            heads, n = list(heads), len(heads)
            if n > max_size:
                continue
            for i in range(n):
                for j in range(i + 1, n):
                    h1, h2 = min(heads[i], heads[j]), max(heads[i], heads[j])
                    pair2w[(h1, h2)] = pair2w.setdefault((h1, h2), 0) + 1

        return pair2w

    @staticmethod
    def count_ent_path(kg, e1, e2):
        """
        count path num from e1 to e2, limit len <= 2
        :param kg:
        :param e1:
        :param e2:
        :return:
        """
        _, hop1_neighbors = kg.out_edges(e1, form='uv')
        _, hop2_neighbors = kg.out_edges(hop1_neighbors, form='uv')

        hop1_npath, hop2_npath = 0, 0
        for n in hop1_neighbors.tolist():
            if n == e2:
                hop1_npath += 1
        for n in hop2_neighbors.tolist():
            if n == e2:
                hop2_npath += 1

        npath = hop1_npath + hop2_npath

        return npath

    @staticmethod
    def split_data(data: list, bounds: list):
        """
        Split the data into different ranges according to the bounds specified.
        eg. bounds [0, 10, 20, 30], result in 4 ranges: [0, 10), [10, 20), [20, 30), [30, +∞)
        :param data:
        :param bounds:
        :return:
        """
        cfg = utils.get_global_config()
        i2str = dict()
        for i in range(len(bounds)):
            if i < len(bounds)-1:
                i2str[i] = '[{}, {})'.format(bounds[i], bounds[i+1])
            else:
                if cfg.dataset == 'FB15k_237':
                    i2str[i] = '[{}, +∞)'.format(bounds[i])
                elif cfg.dataset == 'WN18RR':
                    i2str[i] = '[{}, Max]'.format(bounds[i])
                else:
                    raise NotImplementedError
        range2ndata = dict([(i2str[i], 0) for i in range(len(bounds))])
        # record the data's id
        range2id = dict([(i2str[i], []) for i in range(len(bounds))])
        for i, d in enumerate(data):
            assert d >= bounds[0]
            # find which range the date belongs, from the back forward.
            for j in range(len(bounds) - 1, -1, -1):
                if d >= bounds[j]:
                    range2ndata[i2str[j]] += 1
                    range2id[i2str[j]].append(i)
                    break
        return range2ndata, range2id


def draw_fb_baselines():
    """
    Figure of correlation between SE metrics and model prediction rank (paper Figure 2).
    :return:
    """
    cfg = utils.get_global_config()
    met = Metrics()
    dir_path = join(cfg.project_dir, 'drawing', 'FB15k_237')
    se_metric_path = join(dir_path, 'data2metrics')

    # compute SE metrics
    if exists(se_metric_path):
        logging.info('SE metrics has been computed')
    else:
        logging.info('compute SE metrics...')
        met.compute_semantic_metrics(save_path=se_metric_path)
        logging.info('computing end')

    logging.info('draw SE - Rank fig')
    rank_paths = [join(dir_path, 'ConvE_rank'), join(dir_path, 'CompGCN_rank'),
                  join(dir_path, 'RotatE_rank'), join(dir_path, 'TransE_rank'),
                  join(dir_path, 'DistMult_rank'), join(dir_path, 'ComplEx_rank')]
    met.plot_se2rank_6bar(
        model_name=['ConvE', 'CompGCN', 'RotatE', 'TransE', 'DistMult', 'ComplEx'],
        rank_load_paths=rank_paths, fig_save_path=join(dir_path, 'fb_baselines_se2rank.svg'),
        se_metric_path=se_metric_path
    )
    logging.info('drawing end and fig saved')


def draw_wn_baselines():
    """
    Figure of correlation between SE metrics and model prediction rank (paper Figure 6).
    :return:
    """
    cfg = utils.get_global_config()
    met = Metrics()
    dir_path = join(cfg.project_dir, 'drawing', 'WN18RR')
    se_metric_path = join(dir_path, 'data2metrics')

    # Figure 6
    # compute SE metrics
    if exists(se_metric_path):
        logging.info('SE metrics has been computed')
    else:
        logging.info('compute SE metrics...')
        met.compute_semantic_metrics(save_path=se_metric_path)
        logging.info('computing end')

    logging.info('draw SE - Rank fig')
    rank_paths = [join(dir_path, 'ComplEx_rank'), join(dir_path, 'DistMult_rank'),
                  join(dir_path, 'ConvE_rank'), join(dir_path, 'CompGCN_rank'),
                  join(dir_path, 'RotatE_rank'), join(dir_path, 'TransE_rank')]
    met.plot_se2rank_6bar(
        model_name=['ComplEx', 'DistMult', 'ConvE', 'CompGCN', 'RotatE', 'TransE'],
        rank_load_paths=rank_paths, fig_save_path=join(dir_path, 'wn_baselines_se2rank.svg'),
        se_metric_path=se_metric_path
    )
    logging.info('drawing end and fig saved')


@hydra.main(config_path=join('..', 'config'), config_name="config")
def main(cfg: DictConfig):
    utils.set_global_config(cfg)

    if cfg.dataset == 'FB15k_237':
        draw_fb_baselines()
    elif cfg.dataset == 'WN18RR':
        draw_wn_baselines()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
