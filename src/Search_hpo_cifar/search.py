import os
import sys
import time
import glob
import numpy as np
import pickle
import torch
import logging
import argparse
import torch
import random


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

from network import ShuffleNetV2_OneShot

from tester import get_cand_err, test_nni

from flops import get_cand_flops

from torch.autograd import Variable
import collections
import sys
sys.setrecursionlimit(10000)
import argparse

import functools
print = functools.partial(print, flush=True)

choice = lambda x: x[np.random.randint(len(x))] if isinstance(x, tuple) else choice(tuple(x))

class EvolutionSearcher(object):

    def __init__(self, args):
        self.args = args

        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.flops_limit = args.flops_limit

        # search network
        # self.model = ShuffleNetV2_OneShot()
        # self.model = ShuffleNetV2_OneShot(n_class=100)
        from network_origin import cifar_fast
        self.model = cifar_fast(input_size=32, n_class=100)

        self.model = torch.nn.DataParallel(self.model).cuda()

        # supernet_state_dict = torch.load(
        #     '../Supernet/models/checkpoint-latest.pth.tar')['state_dict']

        # /home/ubuntu/0_datasets/Search
        # supernet_state_dict = torch.load(
        #     '/home/ubuntu/0_datasets/Supernet/checkpoint-150000.pth.tar')['state_dict']

        # one path
        # supernet_state_dict = torch.load(
        #     '/home/ubuntu/2_workspace/nni_hpo/3rdparty/SinglePathOneShot/src/Supernet_hpo_cifar/models_one_path/checkpoint-latest.pth.tar')['state_dict']

        # four
        # supernet_state_dict = torch.load(
        #     '/home/ubuntu/2_workspace/nni_hpo/3rdparty/SinglePathOneShot/src/Supernet_hpo_cifar/models_fixed_lr_6076/checkpoint-latest.pth.tar')['state_dict']


        # supernet_state_dict = torch.load(
        #     '/home/ubuntu/2_workspace/nni_hpo/3rdparty/SinglePathOneShot/src/Supernet_hpo_cifar/models/checkpoint-000200.pth.tar')['state_dict']

        ######

        # layerwise
        # supernet_state_dict = torch.load(
        #     '/home/ubuntu/2_workspace/nni_hpo/3rdparty/SinglePathOneShot/src/Supernet_hpo_cifar/models/layer-wise5e-4-checkpoint-000020.pth.tar')['state_dict']

        # global hpo
        # supernet_state_dict = torch.load(
        #     '/home/ubuntu/2_workspace/nni_hpo/3rdparty/SinglePathOneShot/src/Supernet_hpo_cifar/models/70-25-hpo-layer-wisecheckpoint-latest.pth.tar')['state_dict']

        supernet_state_dict = torch.load(
            '/home/ubuntu/2_workspace/nni_hpo/3rdparty/SinglePathOneShot/src/Supernet_hpo_cifar/models/71.9150641025641-hpo-layer-wisecheckpoint-latest.pth.tar')['state_dict']

        # global hpo + layerwise
        # supernet_state_dict = torch.load(
        #     '/home/ubuntu/2_workspace/nni_hpo/3rdparty/SinglePathOneShot/src/Supernet_hpo_cifar/models/layer-wise5e-4-checkpoint-000020.pth.tar')['state_dict']

        self.model.load_state_dict(supernet_state_dict)

        self.log_dir = args.log_dir
        self.checkpoint_name = os.path.join(self.log_dir, 'checkpoint.pth.tar')

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []

        self.nr_layer = 7 # 20
        self.nr_state = 4 # 4

    def save_checkpoint(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        info = {}
        info['memory'] = self.memory
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        torch.save(info, self.checkpoint_name)
        print('save checkpoint to', self.checkpoint_name)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        info = torch.load(self.checkpoint_name)
        self.memory = info['memory']
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        print('load checkpoint from', self.checkpoint_name)
        return True

    def is_legal(self, cand):
        assert isinstance(cand, tuple) and len(cand) == self.nr_layer
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        if 'visited' in info:
            return False

        # 统计flops
        # if 'flops' not in info:
        #     info['flops'] = get_cand_flops(cand)
        # print(cand, info['flops'])

        # 候选flop limited
        # if info['flops'] > self.flops_limit:
        #     print('flops limit exceed')
        #     return False

        info['err'] = get_cand_err(self.model, cand, self.args)
        # info['err'] = test_nni(self.model, cand, self.args)


        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random(self, num):
        print('random select ........')
        cand_iter = self.stack_random_cand(lambda: tuple(np.random.randint(self.nr_state) for i in range(self.nr_layer)))
        # 取出指定数量的随机候选, population_num
        while len(self.candidates) < num:
            cand = next(cand_iter)
            # 候选的限制
            if not self.is_legal(cand):
                continue

            self.candidates.append(cand)
            print('random {}/{}: {} '.format(len(self.candidates), num, self.candidates))
        print('random_num = {}'.format(len(self.candidates)))

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(choice(self.keep_top_k[k]))
            for i in range(self.nr_layer):
                if np.random.random_sample() < m_prob:
                    cand[i] = np.random.randint(self.nr_state)
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}: '.format(len(res), mutation_num) + str(cand))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            return tuple(choice([i, j]) for i, j in zip(p1, p2))
        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}: '.format(len(res), crossover_num) + str(cand))

        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        print('population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
            self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        # 续点训练
        # self.load_checkpoint()
        # 拿到初始候选 self.candidates)
        self.get_random(self.population_num)
        # 迭代轮数
        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['err'])

            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['err'])

            print('epoch = {} : top {} result'.format(self.epoch, len(self.keep_top_k[50])))

            for i, cand in enumerate(self.keep_top_k[50]):
                print('No.{} {} Top-1 err = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['err']))
                ops = [i for i in cand]
                print(ops)
            # 带概率
            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)

            # 交叉
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            # 剩余随机？
            self.get_random(self.population_num)

            self.epoch += 1

        self.save_checkpoint()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=str, default='log_cifar')
    parser.add_argument('--flops-limit', type=float, default=330 * 1e6)
    parser.add_argument('--max-train-iters', type=int, default=5) # 200 fintune 参数子网
    parser.add_argument('--max-test-iters', type=int, default=3) # 40
    parser.add_argument('--train-batch-size', type=int, default=128)
    parser.add_argument('--test-batch-size', type=int, default=100) # 200

    parser.add_argument('--m_prob', type=float, default=0.1)
    parser.add_argument('--max-epochs', type=int, default=4) # 20
    parser.add_argument('--population-num', type=int, default=10) # 50
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--crossover-num', type=int, default=5) # 25
    parser.add_argument('--mutation-num', type=int, default=5) # 25
    args = parser.parse_args()

    t = time.time()

    searcher = EvolutionSearcher(args)

    searcher.search()

    print('total searching time = {:.2f} hours'.format(
        (time.time() - t) / 3600))

if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)
