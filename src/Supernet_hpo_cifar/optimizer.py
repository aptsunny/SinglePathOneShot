# import torch
import torch.optim as optim
# from utils import get_parameters, fast_hpo_lr_parameters

# weight-decay
def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []

    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)

    assert len(list(model.parameters())) == len(
        group_weight_decay) + len(group_no_weight_decay)

    groups = [dict(params=group_weight_decay),
              dict(params=group_no_weight_decay, weight_decay=0.)]

    return groups

def fast_hpo_lr_parameters(model, lr_group, arch_search=None):
    rest_name = []
    figure_ = []
    for name, param in model.named_parameters():
        rest_name.append(param)
        figure_.append(name)

    figure_choice=[]
    choice = []
    if len(rest_name) == 128:
        loc = 0
        # 1+(3+3+6+6)*7+1
        # conv3/conv5/dwconv3/dwconv5
        split_list = [1,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      1]
        for i in range(0, len(split_list), 1):
            st = loc
            en = loc + split_list[i]
            # print(loc, loc + split_list[i])
            b = figure_[st:en]
            a = rest_name[st:en]
            loc = en
            figure_choice.append(b)
            choice.append(a)

    groups = [dict(params=choice[x], lr=lr_group[x]) for x in range(len(choice))]
    return groups

def combined(model, lr_group, arch_search=None):
    rest_name = []
    figure_ = []
    for name, param in model.named_parameters():
        rest_name.append(param)
        figure_.append(name)

    figure_choice=[]
    choice = []
    if len(rest_name) == 128:
        loc = 0
        # 1+ (3+3+6+6)*7+1
        split_list = [1,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      3, 3, 6, 6,
                      1]
        for i in range(0, len(split_list), 1):
            st = loc
            en = loc + split_list[i]
            # print(loc, loc + split_list[i])
            b = figure_[st:en]
            a = rest_name[st:en]
            loc = en
            figure_choice.append(b)
            choice.append(a)

    groups = [dict(params=choice[x], lr=lr_group[x]) for x in range(len(choice))]
    return groups


def select_optim(args, lr_hpo):
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(lr_hpo,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    if args.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(lr_hpo,
                              lr=args.learning_rate)
    if args.optimizer  == 'Adagrad':
        optimizer = optim.Adagrad(lr_hpo,
                              lr=args.learning_rate)
    if args.optimizer  == 'Adam':
        optimizer = optim.Adam(lr_hpo,
                              lr=args.learning_rate)
    return optimizer

def get_optim(args, model):
    if args.layerwise_lr:
        # 128
        # lr_group = [args['prep'],
        #             args['layer1_conv0_3_3'],
        #             args['layer1_conv1_3_3'],
        #             args['layer1_conv2_3_3'],
        #             args['layer2_conv0_3_3'],
        #             args['layer3_conv0_3_3'],
        #             args['layer3_conv1_3_3'],
        #             args['layer3_conv2_3_3'],
        #             args['rest']]

        # lr_group = [0.1]* 30

        # 4e-5

        lr_3 = [
            0.01866,
            0.02587,
            0.01747,
            0.02627,
            0.00441,
            0.00072,
            0.01174

        ]
        lr_5 = [
            0.02597,
            0.00556,
            0.03097,
            0.05775,
            0.00104,
            0.01952,
            0.00009

        ]
        lr_d3 = [
            0.08191,
            0.02992,
            0.04292,
            0.00163,
            0.01491,
            0.00177,
            0.0005

        ]
        lr_d5 = [
            0.09836,
            0.09947,
            0.00578,
            0.08196,
            0.09992,
            0.02599,
            0.00012

        ]

        lr_group = [] #
        lr_group.append(0.00051)  #
        for i in range(7):
            lr_group.append(lr_3[i])
            lr_group.append(lr_5[i])
            lr_group.append(lr_d3[i])
            lr_group.append(lr_d5[i])
        lr_group.append(0.00094)  #

        # print(len(lr_group))

        optimizer = select_optim(args, fast_hpo_lr_parameters(model, lr_group))
        # optimizer = select_optim(args, combined(model, lr_group))

    elif args.global_lr:
        optimizer = select_optim(args, get_parameters(model))
        # optimizer = torch.optim.SGD(get_parameters(model),
        #                             lr=args.learning_rate,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay)

    else:
        optimizer = select_optim(args, get_parameters(model))
        # optimizer = torch.optim.SGD(get_parameters(model),
        #                             lr=args.learning_rate,
        #                             momentum=args.momentum,
        #                             weight_decay=args.weight_decay)

    return optimizer