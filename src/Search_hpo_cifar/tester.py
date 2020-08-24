import torch

# from imagenet_dataset import get_train_dataprovider, get_val_dataprovider
# from dataset import get_cifar_train_dataprovider, get_cifar_val_dataprovider
from torch.utils.data import SubsetRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
import dataset as dataset_cifar
from dataset import DataIterator, SubsetSampler, OpencvResize, ToBGRTensor
import torch.nn as nn
import tqdm

assert torch.cuda.is_available()

train_dataprovider, val_dataprovider = None, None


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func

def get(args):
    # load dataset
    dataset_train, dataset_valid = dataset_cifar.get_dataset("cifar100")
    # print(len(dataset_train), len(dataset_valid))

    split = 0.0
    split_idx = 0
    train_sampler = None
    if split > 0.0:
        sss = StratifiedShuffleSplit(n_splits=5, test_size=split, random_state=0)
        sss = sss.split(list(range(len(dataset_train))), dataset_train.targets)
        for _ in range(split_idx + 1):
            train_idx, valid_idx = next(sss)
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetSampler(valid_idx)
    else:
        valid_sampler = SubsetSampler([])

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.train_batch_size, shuffle=True if train_sampler is None else False, num_workers=32,
        pin_memory=True, sampler=train_sampler, drop_last=True) # 32

    valid_loader = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.test_batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True) # 16

    train_dataprovider = DataIterator(train_loader)
    val_dataprovider = DataIterator(valid_loader)
    # args.test_interval = len(valid_loader)
    # args.val_interval = int(len(dataset_train) / args.batch_size)  # step
    print('load valid dataset successfully, and images:{}, batchsize:{}, step:{}'.format(len(dataset_valid), args.test_batch_size, len(dataset_valid)//args.test_batch_size))
    return train_dataprovider, val_dataprovider, len(dataset_valid)//args.test_batch_size

@no_grad_wrapper
def get_cand_err(model, cand, args):
    # global train_dataprovider, val_dataprovider
    train_dataprovider, val_dataprovider, steps = get(args)
    """
    if train_dataprovider is None:
        use_gpu = False
        train_dataprovider = get_train_dataprovider(
            args.train_batch_size, use_gpu=False, num_workers=8)
        val_dataprovider = get_val_dataprovider(
            args.test_batch_size, use_gpu=False, num_workers=8)
    """

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    max_train_iters = args.max_train_iters
    max_test_iters = args.max_test_iters

    """
    print('clear bn statics....')
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = torch.zeros_like(m.running_mean)
            m.running_var = torch.ones_like(m.running_var)
    print('train bn with training set (BN sanitize) ....')
    
    model.train()

    for step in tqdm.tqdm(range(max_train_iters)):
        # print('train step: {} total: {}'.format(step,max_train_iters))
        data, target = train_dataprovider.next()
        # print('get data',data.shape)

        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)

        output = model(data, cand)
        # 无反向传播
        del data, target, output
    """

    ######
    top1 = 0
    top5 = 0
    total = 0
    print('starting test....')
    model.eval()

    # for step in tqdm.tqdm(range(max_test_iters)):
    for step in tqdm.tqdm(range(steps)):
        # print('test step: {} total: {}'.format(step,max_test_iters))
        data, target = val_dataprovider.next()
        batchsize = data.shape[0]
        # print('get data',data.shape)
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        # print(len(data), cand)

        logits = model(data, cand)

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        # print(prec1.item(),prec5.item())
        top1 += prec1.item() * batchsize
        top5 += prec5.item() * batchsize
        total += batchsize
        del data, target, logits, prec1, prec5

    top1, top5 = top1 / total, top5 / total
    top1, top5 = 1 - top1 / 100, 1 - top5 / 100
    print('top1 acc: {:.2f} top5 acc: {:.2f}'.format(100- top1 * 100, 100- top5 * 100))
    return top1, top5
    """

    testloader = val_dataprovider
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    test_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        # for batch_idx, (inputs, targets) in enumerate(testloader):
        # for batch_idx in range(1, batches + 1):
        for batch_idx in range(max_test_iters):
            # print(batch_idx, batches)
            inputs, targets = testloader.next()
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs, cand)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100. * correct / total
            print(acc)

            # progress_bar(cand, batch_idx, max_test_iters,
            #              'Loss: %.3f | Acc: %.3f  (%d/%d)' % (test_loss / (batch_idx + 1), acc, correct, total))

            # progress_bar(batch_idx, batches,
            #              'Loss: %.3f | Acc: %.3f  (%d/%d)' % (sum_test_loss / (batch_idx + 1), acc, correct, total))
            # if batches > 0 and (batch_idx+1) >= batches:
            #     pass

    # Save checkpoint.
    # acc = 100. * correct / total
    # if acc > best_acc:
    if acc > best_acc:
        best_acc = acc
    return acc, best_acc
    """


def test_nni(net, architecture, args):
    global best_acc

    # net = nn.DataParallel(net)
    device = torch.device("cuda")
    criterion = CrossEntropyLabelSmooth(100, 0.1).cuda()
    _, testloader, batches = get(args)
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
            print(inputs.shape, targets.shape)

            outputs = net(inputs, architecture)
            print(len(targets), len(outputs))

            loss = criterion(outputs, targets)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.*correct/total
            print(acc)

            # progress_bar(architecture, batch_idx, batches, 'Loss: %.3f | Acc: %.3f  (%d/%d)' % (test_loss/(batch_idx+1), acc, correct, total))

            # progress_bar(batch_idx, batches,
            #              'Loss: %.3f | Acc: %.3f  (%d/%d)' % (sum_test_loss / (batch_idx + 1), acc, correct, total))
            # if batches > 0 and (batch_idx+1) >= batches:
            #     pass

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    return acc, best_acc

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * \
            targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    pass
