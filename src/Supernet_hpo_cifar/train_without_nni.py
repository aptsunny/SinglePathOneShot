from __future__ import print_function

import os
import numpy as np
import time
import logging
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import dataset as dataset_cifar
from utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, progress_bar, adjust_bn_momentum

from flops import get_cand_flops
from optimizer import get_optim

import nni

_logger = logging.getLogger("cifar100_train_Supernet_hpo_layer-wise")

def train_nni(args, net, device, epoch, batches=-1):
    optimizer = args.optimizer
    criterion = args.loss_function
    scheduler = args.scheduler
    trainloader = args.train_dataprovider

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    # scheduler.step()

    for batch_idx in range(batches):
        if args.bn_process:
            adjust_bn_momentum(net, args.all_iters, batch_idx)

        args.all_iters += 1

        inputs, targets = trainloader.next()
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if args.constrain_flops:
            get_random_cand = lambda:tuple(np.random.randint(4) for i in range(20)) # 4
            flops_l, flops_r, flops_step = 290, 360, 10
            bins = [[i, i+flops_step] for i in range(flops_l, flops_r, flops_step)]

            def get_uniform_sample_cand(*,timeout=500):
                idx = np.random.randint(len(bins))
                l, r = bins[idx]
                for i in range(timeout):
                    cand = get_random_cand()
                    if l*1e6 <= get_cand_flops(cand) <= r*1e6:
                        # print("the {} iters is {}.\n".format(iters, cand))
                        return cand
                return get_random_cand()
            outputs = net(inputs, get_uniform_sample_cand())
        else:
            # get_random_cand = lambda: tuple(np.random.randint(1) for i in range(7))
            get_random_cand = lambda: tuple(np.random.randint(4) for i in range(7))  # 4
            architecture = get_random_cand()
            outputs = net(inputs, architecture)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        # print('current lr group:', scheduler.get_last_lr())
        scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        progress_bar(architecture, batch_idx, batches, 'Loss: %.3f | Acc: %.3f (%d/%d)' % (train_loss/(batch_idx+1), acc, correct, total))

        loss_output = train_loss/(batch_idx+1)

        # final step
        if batches > 0 and (batch_idx+1) >= batches:
            return loss_output, acc

def test_nni(args, net, device, epoch, batches=-1):
    best_acc = args.best_acc
    testloader = args.val_dataprovider
    criterion = args.loss_function
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        # for batch_idx, (inputs, targets) in enumerate(testloader):
        # for batch_idx in range(1, batches + 1):
        for batch_idx in range(batches):
            # print(batch_idx, batches)
            inputs, targets = testloader.next()
            inputs, targets = inputs.to(device), targets.to(device)

            # get_random_cand = lambda: tuple(np.random.randint(1) for i in range(7))  # 4
            # outputs = net(inputs, get_random_cand())

            # get_random_cand = lambda: tuple(np.random.randint(4) for i in range(7))  # 4
            # architecture = get_random_cand()

            # 测试固定网络结构/ 测试某个范围下的网络结构/ 符合某些要求的网络结构
            get_random_cand = lambda: tuple(np.random.randint(1)+ int(total/128%4) for i in range(7))
            architecture = get_random_cand()
            outputs = net(inputs, architecture)

            # print(len(targets), len(outputs))
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.*correct/total
            # print(acc)

            progress_bar(architecture, batch_idx, batches, 'Loss: %.3f | Acc: %.3f  (%d/%d)' % (test_loss/(batch_idx+1), acc, correct, total))

            # progress_bar(batch_idx, batches,
            #              'Loss: %.3f | Acc: %.3f  (%d/%d)' % (sum_test_loss / (batch_idx + 1), acc, correct, total))
            # if batches > 0 and (batch_idx+1) >= batches:
            #     pass

    # Save checkpoint.
    acc = 100.*correct/total

    if acc > best_acc:
        best_acc = acc

    if epoch % args.save_interval == 0:
        print('Saving..')
        """
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        """
        save_checkpoint({
            'state_dict': net.state_dict(),},
            epoch, tag='{}-hpo-layer-wise'.format(best_acc)
        )
    return acc, best_acc

def prepare(args, RCV_CONFIG):
    args.momentum = RCV_CONFIG['momentum']
    args.bn_process = True if RCV_CONFIG['bn_process']=='True' else False
    args.learning_rate = RCV_CONFIG['learning_rate']
    args.weight_decay = RCV_CONFIG['weight_decay']
    args.label_smooth = RCV_CONFIG['label_smooth']
    args.lr_scheduler = RCV_CONFIG['lr_scheduler']
    args.randAugment = True if RCV_CONFIG['randAugment'] == 'True' else False


    # if RCV_CONFIG['momentum'] == 'vgg':
    #     net = VGG('VGG19')
    # if RCV_CONFIG['model'] == 'resnet18':
    #     net = ResNet18()
    # if RCV_CONFIG['model'] == 'googlenet':
    #     net = GoogLeNet()

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True

    if args.cifar100:
        # train_dataprovider, val_dataprovider, train_step, valid_step = dataset_cifar.get_dataset("cifar100", batch_size=args.batch_size, RandA=args.randAugment)

        train_dataprovider, val_dataprovider, train_step, valid_step = dataset_cifar.get_dataset("cifar10",
                                                                                                 batch_size=args.batch_size,
                                                                                                 RandA=args.randAugment)
        print('load data successfully')
    else:
        assert os.path.exists(args.train_dir)
        from dataset import DataIterator, SubsetSampler, OpencvResize, ToBGRTensor
        train_dataset = datasets.ImageFolder(
            args.train_dir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(0.5),
                ToBGRTensor(),
            ])
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=1, pin_memory=use_gpu)
        train_dataprovider = DataIterator(train_loader)

        assert os.path.exists(args.val_dir)
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(args.val_dir, transforms.Compose([
                OpencvResize(256),
                transforms.CenterCrop(224),
                ToBGRTensor(),
            ])),
            batch_size=200, shuffle=False,
            num_workers=1, pin_memory=use_gpu
        )
        val_dataprovider = DataIterator(val_loader)
        print('load data successfully')

    # Imagenet
    # from network import ShuffleNetV2_OneShot
    # model = ShuffleNetV2_OneShot(n_class=1000)

    # Special for cifar
    from network_origin import cifar_fast
    model = cifar_fast(input_size=32, n_class=100)

    # Optimizer
    optimizer = get_optim(args, model)

    # Label Smooth
    if args.label_smooth > 0:
        criterion = CrossEntropyLabelSmooth(100, args.label_smooth)
    else:
        # print('CrossEntropyLoss')
        criterion = nn.CrossEntropyLoss()


    if args.lr_scheduler == 'Lambda':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                        lambda step : (1.0-step/(args.epochs * train_step)) if step <= (args.epochs * train_step) else 0, last_epoch=-1)
    elif args.lr_scheduler == 'Cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8, last_epoch=-1)


    if use_gpu:
        model = nn.DataParallel(model)
        cudnn.benchmark = True
        loss_function = criterion.cuda()
        device = torch.device("cuda")
    else:
        loss_function = criterion
        device = torch.device("cpu")
    model = model.to(device)

    args.optimizer = optimizer
    args.loss_function = loss_function
    args.scheduler = scheduler
    args.train_dataprovider = train_dataprovider
    args.val_dataprovider = val_dataprovider
    args.best_acc = 0.0
    args.all_iters = 1

    return model, device, train_step, valid_step


def get_args():
        parser = argparse.ArgumentParser("Train Supernet HPO")
        parser.add_argument('--eval', default=False, action='store_true')
        parser.add_argument('--eval-resume', type=str, default='./snet_detnas.pkl', help='path for eval model')
        parser.add_argument('--total-iters', type=int, default=15000, help='total iters')  # 150000
        parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
        parser.add_argument('--auto-continue', type=bool, default=False, help='report frequency')  # True
        parser.add_argument('--display-interval', type=int, default=20, help='report frequency')  # 20
        parser.add_argument('--val-interval', type=int, default=10000, help='report frequency')
        parser.add_argument('--save-interval', type=int, default=10, help='report frequency')  # 10000
        parser.add_argument('--train-dir', type=str, default='data/train', help='path to training dataset')
        parser.add_argument('--val-dir', type=str, default='data/val', help='path to validation dataset')
        parser.add_argument('--cifar100', default=True, action='store_true')

        parser.add_argument('--epochs', type=int, default=400, help='total epochs')  # 200
        parser.add_argument('--batch-size', type=int, default=128, help='batch size')  # 1024

        # parser.add_argument('--learning-rate', type=float, default=0.1, help='init learning rate')  # 0.5
        parser.add_argument('--global-lr', default=False, action='store_true')
        parser.add_argument('--layerwise-lr', default=False, action='store_true', help='True/False')

        parser.add_argument('--optimizer', type=str, default='SGD',
                            help='optimizer:SGD/Adadelta/Adam   Adagrad/->cpu')

        # parser.add_argument('--momentum', type=float, default=0.9, help='momentum, special for SGD')
        # parser.add_argument('--bn-process', default=False, action='store_true', help='adjust_bn_momentum, special for SGD')
        # parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay, 4e-5/5e-4, special for SGD')


        # parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing')
        # parser.add_argument('--lr-scheduler', type=str, default='Lambda', help='lr-scheduler: Lambda/Cosine')
        # parser.add_argument('--randAugment', default=False, action='store_true')

        parser.add_argument('--constrain-flops', default=False, action='store_true')

        args = parser.parse_args()
        return args

if __name__ == "__main__":
    args = get_args()

    try:

        # RCV_CONFIG = nni.get_next_parameter()

        """
        # 69.93
        RCV_CONFIG = {
            "learning_rate": 0.06078498555747216,
            "weight_decay": 0.00031250944304909474,
            "momentum": 0.9,
            "bn_process": "False",
            "label_smooth": 0,
            "lr_scheduler": "Lambda",
            "randAugment": "False"
                      }
        
        # 70.25
        RCV_CONFIG = {
            "learning_rate": 0.04356903098698344,
            "weight_decay": 0.00033451626443547404,
            "momentum": 0.9,
            "bn_process": "False",
            "label_smooth": 0,
            "lr_scheduler": "Lambda",
            "randAugment": "False"
        }
        """
        # 70.58
        RCV_CONFIG = {
            "learning_rate": 0.05729618712189671,
            "weight_decay": 0.000477834597976567,
            "momentum": 0.85,
            "bn_process": "False",
            "label_smooth": 0.1,
            "lr_scheduler": "Lambda",
            "randAugment": "False"
        }


        # _logger.debug(RCV_CONFIG)

        model, device, train_step, valid_step = prepare(args, RCV_CONFIG)
        start_epoch = 1
        for epoch in range(start_epoch, start_epoch + args.epochs):
            loss_output, train_acc = train_nni(args, model, device, epoch, train_step)
            acc, best_acc = test_nni(args, model, device, epoch, valid_step)

            print(
                'Epoch {}, loss/train acc = {:.2f}/{:.2f}, \
                val acc/best acc = {:.2f}/{:.2f},'.format(epoch, loss_output, train_acc, acc, best_acc))
            """
            nni.report_intermediate_result(acc)
        nni.report_final_result(best_acc)
        """

    except Exception as exception:
        _logger.exception(exception)
        raise


