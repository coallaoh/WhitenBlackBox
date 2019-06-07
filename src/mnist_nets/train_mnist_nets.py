#!/usr/bin/env python

__author__ = 'joon'

import sys

sys.path.insert(0, 'src')

from mnist_nets.mnist import training as mnist_train
from util.construct_controls import apply_explist
from util.exceptions import CacheFileExists


def load_conf_control_models(archset):
    conf = dict(
        exp_phase='modelzoo-mnist',
        save=True,
        overridecache_model=True,
        overridecache_output=True,
        test_batch_size=1000,
        log_interval=100,
        gpu=None,
    )

    nmodel = int(archset[4:])
    control = dict(
        net=dict(
            name='mnet',
            act=None,
            drop=None,
            pool=None,
            ks=None,
            n_conv=None,
            n_fc=None,
        ),
        opt=dict(
            optimiser=None,
            momentum=0.5,
            lr=0.1,
            epochs=100,
            batch_size=None,
        ),
        data=dict(
            dataset='mnist',
            subset=None,
        ),
        seed=None,
    )
    attr_list = dict(
        act=['relu', 'elu', 'prelu', 'tanh'],
        drop=['normal', 'none'],
        pool=['max_2', 'none'],
        ks=[3, 5],
        n_conv=[2, 3, 4],
        n_fc=[2, 3, 4],
        optimiser=['SGD', 'ADAM', 'RMSprop'],
        batch_size=[256, 128, 64],
        subset=['all', 'half_0', 'half_1', 'quarter_0', 'quarter_1', 'quarter_2', 'quarter_3'],
        seed=range(1000),
    )
    attr_list_keys = list(attr_list.keys())
    attr_list_list = [attr_list[ky] for ky in attr_list_keys]
    import itertools, random
    all_comb = list(itertools.product(*attr_list_list))
    random.seed(100)
    random.shuffle(all_comb)

    selected_comb = all_comb[:nmodel]

    models = []

    for comb in selected_comb:
        model = dict(
            net=dict(
                act=comb[attr_list_keys.index('act')],
                drop=comb[attr_list_keys.index('drop')],
                pool=comb[attr_list_keys.index('pool')],
                ks=comb[attr_list_keys.index('ks')],
                n_conv=comb[attr_list_keys.index('n_conv')],
                n_fc=comb[attr_list_keys.index('n_fc')]
            ),
            opt=dict(
                optimiser=comb[attr_list_keys.index('optimiser')],
                batch_size=comb[attr_list_keys.index('batch_size')],
            ),
            data=dict(
                subset=comb[attr_list_keys.index('subset')],
            ),
            seed=comb[attr_list_keys.index('seed')])
        model['opt']['lr'] = 0.1 if (model['opt']['optimiser'] == 'SGD') else 0.001
        models.append(model)

    def show_stats():
        stats = {}
        for ky in attr_list_keys:
            subkeys = attr_list[ky]
            stats[ky] = {}
            for sky in subkeys:
                stats[ky][sky] = 0

        for comb in selected_comb:
            for ky in attr_list_keys:
                stats[ky][comb[attr_list_keys.index(ky)]] += 1

        print("******** stats ********")

        for ky in attr_list_keys:
            print(ky)
            subkeys = attr_list[ky]
            for sky in subkeys:
                stats[ky][sky] /= (float(nmodel) / 100)

            print(subkeys)
            print(stats[ky])

        print("***********************")

    show_stats()

    return conf, control, models


def main():
    conf, control, models = load_conf_control_models('dnet10000')
    import copy

    i = 0
    for model_config in models:
        i += 1
        print("#########################")
        print("Model %d/%d being trained" % (i, len(models)))
        print("#########################")
        control_ = copy.deepcopy(control)
        apply_explist(control_, model_config)
        try:
            mnist_train(control_, conf)
        except CacheFileExists:
            print("Model exists; skip this model")
            continue


if __name__ == "__main__":
    main()
