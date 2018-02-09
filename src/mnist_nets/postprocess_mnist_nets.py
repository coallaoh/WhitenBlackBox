#!/usr/bin/env python
from __future__ import print_function

__author__ = 'joon'

import sys

sys.path.insert(0, 'src')

from imports.basic_modules import *
from imports.ResearchTools import *


def exact_configs(cos):
    tokens_noseed = []
    default_noseed = copy.deepcopy(cos[0]._default)
    default_noseed.pop('seed')
    for co in cos:
        ct_noseed = copy.deepcopy(co.control)
        ct_noseed.pop('seed')
        co_noseed = experiment_control(ct_noseed, copy.deepcopy(co.conf), default_noseed)
        tokens_noseed.append(co_noseed.token)

    return tokens_noseed


def acc_distribution(cos):
    for ii, co in enumerate(cos):
        statoutputloc = co.conf['loc']['statoutput']
        acc = load_from_cache(statoutputloc)['acc']
        co.acc = acc


def construct_cos(cos):
    from train_mnist_nets import load_conf_control_models
    from util.construct_controls import apply_explist
    from mnist import config as config_mnet_train

    _, trainnet_control, trainnet_models = load_conf_control_models('dnet10000')
    for i, model_config in enumerate(trainnet_models):
        # print("%d,%d" % (i, len(trainnet_models)))
        trainnet_control_ = copy.deepcopy(trainnet_control)
        apply_explist(trainnet_control_, model_config)
        conf_ = dict(exp_phase='modelzoo-mnist')
        co = config_mnet_train(trainnet_control_, conf_, called=True)
        cos.append(co)
    acc_distribution(cos)


def get_attr_stats(cos):
    stats = {}
    stats['data_size'] = dict(
        all=np.array([co.acc for co in cos if co.control['data']['subset'] == 'all']),
        half=np.array([co.acc for co in cos if 'half' in co.control['data']['subset']]),
        quarter=np.array([co.acc for co in cos if 'quarter' in co.control['data']['subset']]),
    )
    stats['opt_bs'] = dict(
        large=np.array([co.acc for co in cos if co.control['opt']['batch_size'] == 256]),
        med=np.array([co.acc for co in cos if co.control['opt']['batch_size'] == 128]),
        small=np.array([co.acc for co in cos if co.control['opt']['batch_size'] == 64]),
    )
    stats['opt_opt'] = dict(
        SGD=np.array([co.acc for co in cos if co.control['opt']['optimiser'] == 'SGD']),
        ADAM=np.array([co.acc for co in cos if co.control['opt']['optimiser'] == 'ADAM']),
        RMSprop=np.array([co.acc for co in cos if co.control['opt']['optimiser'] == 'RMSprop']),
    )
    stats['arch_act'] = dict(
        relu=np.array([co.acc for co in cos if co.control['net']['act'] == 'relu']),
        elu=np.array([co.acc for co in cos if co.control['net']['act'] == 'elu']),
        prelu=np.array([co.acc for co in cos if co.control['net']['act'] == 'prelu']),
        tanh=np.array([co.acc for co in cos if co.control['net']['act'] == 'tanh']),
    )
    stats['arch_drop'] = dict(
        drop=np.array([co.acc for co in cos if co.control['net']['drop'] == 'normal']),
        nodrop=np.array([co.acc for co in cos if co.control['net']['drop'] == 'none']),
    )
    stats['arch_ks'] = dict(
        ks3=np.array([co.acc for co in cos if co.control['net']['ks'] == 3]),
        ks5=np.array([co.acc for co in cos if co.control['net']['ks'] == 5]),
    )
    stats['arch_nconv'] = dict(
        nconv2=np.array([co.acc for co in cos if co.control['net']['n_conv'] == 2]),
        nconv3=np.array([co.acc for co in cos if co.control['net']['n_conv'] == 3]),
        nconv4=np.array([co.acc for co in cos if co.control['net']['n_conv'] == 4]),
    )
    stats['arch_nfc'] = dict(
        nfc2=np.array([co.acc for co in cos if co.control['net']['n_fc'] == 2]),
        nfc3=np.array([co.acc for co in cos if co.control['net']['n_fc'] == 3]),
        nfc4=np.array([co.acc for co in cos if co.control['net']['n_fc'] == 4]),
    )
    stats['arch_pool'] = dict(
        maxpool=np.array([co.acc for co in cos if co.control['net']['pool'] == 'max_2']),
        nopool=np.array([co.acc for co in cos if co.control['net']['pool'] == 'none']),
    )

    for ky in stats.keys():
        print(":::: Stats for %10s ::::" % ky)
        print("%7s %7s %7s %7s %7s" % ("Type", "Max", "Median", "Mean", "Min"))
        for item in stats[ky].keys():
            data = stats[ky][item]
            print("%7s\t%2.1f\t%2.1f\t%2.1f\t%2.1f" % (
                item, data.max(), np.median(data), data.mean(), data.min()))

    accs = np.array([co.acc for co in cos])
    cnts, bins = np.histogram(accs, 50, range=(0, 100))
    print("Distribution of accuracies")
    print(bins)
    print(cnts)


def prune_cos(cos, level):
    new_cos = [co for co in cos if co.acc >= level]
    return new_cos


def compute_nparam(cos):
    from mnist import load_model
    for co in cos:
        model = load_model(co.control)
        n_param = 0
        for p in model.parameters():
            n_param += np.array(p.size()).prod()
        if not isinstance(co.control['seed'], int):
            n_param *= len(co.control['seed'])
        co.n_param = n_param


def add_ensembles(cos, cnt_list):
    tokens_noseed = exact_configs(cos)
    vec, idx, inv, cnt = np.unique(tokens_noseed, return_counts=True, return_index=True, return_inverse=True)
    nensembles = []
    for M in cnt_list:
        ccc = 0
        u_inds_M = np.where(cnt >= M)[0]
        for u_ind_m in u_inds_M:
            ens_token = vec[u_ind_m]
            ens_inds_all = np.where(np.array(tokens_noseed) == ens_token)[0]
            ens_inds_subsets = list(itertools.combinations(ens_inds_all, M))
            for ens_inds in ens_inds_subsets:
                co_new = copy.deepcopy(cos[ens_inds[0]])
                co_new.control['seed'] = [cos[ii].control['seed'] for ii in ens_inds]
                co_new.acc = None
                cos.append(co_new)
                ccc += 1
        nensembles.append(ccc)

    print(dict(zip(cnt_list, nensembles)))


def main():
    cos = []
    construct_cos(cos)
    compute_nparam(cos)

    cos = prune_cos(cos, 98)
    get_attr_stats(cos)
    add_ensembles(cos, [2, 3, 4])
    save_to_cache(cos, 'cache/modelzoo-mnist/dnet10000_cos_pruned_ensembled.pkl')


if __name__ == "__main__":
    main()
