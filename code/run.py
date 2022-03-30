#!/usr/bin/python3

import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import utils
from data_helper import TrainDataset, construct_kg, get_kg
import hydra
from omegaconf import DictConfig
from torch.optim.lr_scheduler import LambdaLR
import pickle
from os.path import join
from model_helper import train_step, evaluate
from model import SE_GNN


def save_model(model, save_variables):
    """
    Save the parameters of the model
    :param model:
    :param save_variables:
    :return:
    """
    cfg = utils.get_global_config()
    pickle.dump(cfg, open('config.pickle', 'wb'))

    state_dict = {
        'model_state_dict': model.state_dict(),  # model parameters
        **save_variables
    }

    torch.save(state_dict, 'checkpoint.torch')


def get_linear_scheduler_with_warmup(optimizer, warmup_steps: int, max_steps: int):
    """
    Create scheduler with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        """
        Compute a ratio according to current step,
        by which the optimizer's lr will be mutiplied.
        :param current_step:
        :return:
        """
        assert current_step <= max_steps
        if current_step < warmup_steps:
            return current_step / warmup_steps
        else:
            return (max_steps - current_step) / (max_steps - warmup_steps)

    assert max_steps >= warmup_steps

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def format_metrics(name, h_metric, t_metric):
    msg_h = name + ' (head) - MRR: {:5.4f}, MR: {:7.2f}, H@1: {:4.3f}, H@3: {:4.3f}, H@10: {:4.3f}'
    msg_t = name + ' (tail) - MRR: {:5.4f}, MR: {:7.2f}, H@1: {:4.3f}, H@3: {:4.3f}, H@10: {:4.3f}'
    msg_avg = name + ' (avg) - MRR: {:5.4f}, MR: {:7.2f}, H@1: {:4.3f}, H@3: {:4.3f}, H@10: {:4.3f}'
    msg_h = msg_h.format(h_metric['MRR'], h_metric['MR'],
                         h_metric['HITS@1'], h_metric['HITS@3'], h_metric['HITS@10'])
    msg_t = msg_t.format(t_metric['MRR'], t_metric['MR'],
                         t_metric['HITS@1'], t_metric['HITS@3'], t_metric['HITS@10'])
    msg_avg = msg_avg.format(
        (h_metric['MRR'] + t_metric['MRR']) / 2,
        (h_metric['MR'] + t_metric['MR']) / 2,
        (h_metric['HITS@1'] + t_metric['HITS@1']) / 2,
        (h_metric['HITS@3'] + t_metric['HITS@3']) / 2,
        (h_metric['HITS@10'] + t_metric['HITS@10']) / 2
    )
    return msg_h, msg_t, msg_avg


@hydra.main(config_path=join('..', 'config'), config_name="config")
def main(config: DictConfig):
    utils.set_global_config(config)
    cfg = utils.get_global_config()
    assert cfg.dataset in cfg.dataset_list

    # remove randomness
    utils.remove_randomness()

    # print configuration
    logging.info('\n------Config------\n {}'.format(utils.filter_config(cfg)))

    # backup the code and configuration
    code_dir_path = os.path.dirname(__file__)
    project_dir_path = os.path.dirname(code_dir_path)
    config_dir_path = os.path.join(project_dir_path, 'config')
    hydra_current_dir = os.getcwd()
    logging.info('Code dir path: {}'.format(code_dir_path))
    logging.info('Config dir path: {}'.format(config_dir_path))
    logging.info('Model save path: {}'.format(hydra_current_dir))
    os.system('cp -r {} {}'.format(code_dir_path, hydra_current_dir))
    os.system('cp -r {} {}'.format(config_dir_path, hydra_current_dir))

    device = torch.device(cfg.device)
    model = SE_GNN(cfg.h_dim)
    model = model.to(device)

    # load the knowledge graph
    src, dst, rel, hr2eid, rt2eid = construct_kg('train', directed=False)
    kg = get_kg(src, dst, rel, device)
    kg_out_deg = kg.out_degrees(torch.arange(kg.number_of_nodes(), device=device))
    kg_zero_deg_num = torch.sum(kg_out_deg < 1).to(torch.int).item()
    logging.info('kg # node: {}'.format(kg.number_of_nodes()))
    logging.info('kg # edge: {}'.format(kg.number_of_edges()))
    logging.info('kg # zero deg node: {}'.format(kg_zero_deg_num))

    train_loader = DataLoader(
        TrainDataset('train', hr2eid, rt2eid),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.cpu_worker_num,
        collate_fn=TrainDataset.collate_fn
    )

    logging.info('-----Model Parameter Configuration-----')
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' %
                     (name, str(param.size()), str(param.requires_grad)))

    # set optimizer and scheduler
    n_epoch = cfg.epoch
    single_epoch_step = len(train_loader)
    max_steps = n_epoch * single_epoch_step
    warm_up_steps = int(single_epoch_step * cfg.warmup_epoch)
    init_lr = cfg.learning_rate
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=init_lr
    )
    scheduler = get_linear_scheduler_with_warmup(optimizer, warm_up_steps, max_steps)

    logging.info('Training... total epoch: {0}, step: {1}'.format(n_epoch, max_steps))
    last_improve_epoch = 0
    best_mrr = 0.

    for epoch in range(n_epoch):
        loss_list = []
        for batch_data in train_loader:
            train_log = train_step(model, batch_data, kg, optimizer, scheduler)
            loss_list.append(train_log['loss'])
            # get a new kg, since in previous kg some edges are removed.
            if cfg.rm_rate > 0:
                kg = get_kg(src, dst, rel, device)

        val_metrics = evaluate(model, set_flag='valid', kg=kg)['average']
        if val_metrics['MRR'] > best_mrr:
            best_mrr = val_metrics['MRR']
            save_variables = {
                'best_val_metrics': val_metrics
            }
            save_model(model, save_variables)
            improvement_flag = '*'
            last_improve_epoch = epoch
        else:
            improvement_flag = ''

        val_msg = 'Val - MRR: {:5.4f}, MR: {:7.2f}, H@1: {:4.3f}, H@3: {:4.3f}, H@10: {:4.3f} | '
        val_msg = val_msg.format(
            val_metrics['MRR'], val_metrics['MR'],
            val_metrics['HITS@1'], val_metrics['HITS@3'], val_metrics['HITS@10']
        )

        if improvement_flag == '*':
            test_metrics = evaluate(model, set_flag='test', kg=kg)['average']
            val_msg += 'Test - MRR: {:5.4f} | '.format(test_metrics['MRR'])

        val_msg += improvement_flag

        msg = 'Epoch: {:3d} | Loss: {:5.4f} | '
        msg = msg.format(epoch, np.mean(loss_list))
        msg += val_msg
        logging.info(msg)

        # whether early stopping
        if epoch - last_improve_epoch > cfg.max_no_improve:
            logging.info("Long time no improvenment, stop training...")
            break

    logging.info('Training end...')

    # evaluate train and test set
    # load best model
    checkpoint = torch.load('checkpoint.torch')
    model.load_state_dict(checkpoint['model_state_dict'])

    logging.info('Train metrics ...')
    train_metrics = evaluate(model, 'train', kg=kg)
    train_msg = format_metrics('Train', train_metrics['head_batch'], train_metrics['tail_batch'])
    logging.info(train_msg[0])
    logging.info(train_msg[1])
    logging.info(train_msg[2] + '\n')

    logging.info('Valid metrics ...')
    valid_metrics = evaluate(model, 'valid', kg=kg)
    valid_msg = format_metrics('Valid', valid_metrics['head_batch'], valid_metrics['tail_batch'])
    logging.info(valid_msg[0])
    logging.info(valid_msg[1])
    logging.info(valid_msg[2] + '\n')

    logging.info('Test metrics...')
    test_metrics = evaluate(model, 'test', kg=kg)
    test_msg = format_metrics('Test', test_metrics['head_batch'], test_metrics['tail_batch'])
    logging.info(test_msg[0])
    logging.info(test_msg[1])
    logging.info(test_msg[2] + '\n')

    logging.info('Model save path: {}'.format(os.getcwd()))


if __name__ == '__main__':
    main()
