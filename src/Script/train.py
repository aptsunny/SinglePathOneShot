"""
# Log
log_format = '[%(asctime)s] %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%d %I:%M:%S')
t = time.time()
local_time = time.localtime(t)
if not os.path.exists('./log'):
    os.mkdir('./log')
fh = logging.FileHandler(os.path.join('log/train-{}{:02}{}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
"""


def train(model, device, args, *, val_interval, bn_process=False, all_iters=None):

    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_dataprovider = args.train_dataprovider

    t1 = time.time()
    Top1_err, Top5_err = 0.0, 0.0
    model.train()
    for iters in range(1, val_interval + 1):
        scheduler.step()
        if bn_process:
            adjust_bn_momentum(model, iters)

        all_iters += 1
        d_st = time.time()
        data, target = train_dataprovider.next()
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        data_time = time.time() - d_st

        """
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
        output = model(data, get_uniform_sample_cand())

        """

        # get_random_cand = lambda: tuple(np.random.randint(4) for i in range(7))  # 4
        get_random_cand = lambda: tuple(np.random.randint(1) for i in range(7))  # 4
        # print(get_random_cand())
        output = model(data, get_random_cand())

        loss = loss_function(output, target)
        optimizer.zero_grad()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None and p.grad.sum() == 0:
                p.grad = None

        optimizer.step()
        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        Top1_err += 1 - prec1.item() / 100
        Top5_err += 1 - prec5.item() / 100

        if all_iters % args.display_interval == 0:
            printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(all_iters, scheduler.get_lr()[0], loss.item()) + \
                        'Top-1 err = {:.6f},\t'.format(Top1_err / args.display_interval) + \
                        'Top-5 err = {:.6f},\t'.format(Top5_err / args.display_interval) + \
                        'data_time = {:.6f},\ttrain_time = {:.6f}'.format(data_time, (time.time() - t1) / args.display_interval)
            logging.info(printInfo)
            t1 = time.time()
            Top1_err, Top5_err = 0.0, 0.0

        if all_iters % args.save_interval == 0:
            save_checkpoint({
                'state_dict': model.state_dict(),
                }, all_iters)

    return all_iters

def train_one_epoch(epoch, model, device, args, *, step, bn_process=False, all_iters=None):
    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_dataprovider = args.train_dataprovider

    t1 = time.time()
    Top1_err, Top5_err = 0.0, 0.0
    total = 0
    model.train()
    for iters in range(1, step + 1):
        if bn_process:
            adjust_bn_momentum(model, iters)

        all_iters += 1
        d_st = time.time()
        data, target = train_dataprovider.next()
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        data_time = time.time() - d_st

        """
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
        output = model(data, get_uniform_sample_cand())

        """

        # get_random_cand = lambda: tuple(np.random.randint(4) for i in range(7))  # 4
        get_random_cand = lambda: tuple(np.random.randint(4) for i in range(7))  # 4
        # print(get_random_cand())
        architecture =  get_random_cand()
        output = model(data, architecture)

        loss = loss_function(output, target)
        optimizer.zero_grad()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None and p.grad.sum() == 0:
                p.grad = None

        optimizer.step()
        prec1, prec5 = accuracy(output, target, topk=(1, 5))


        total += target.size(0)

        Top1_err += 1 - prec1.item() / 100
        Top5_err += 1 - prec5.item() / 100

        # if all_iters % args.display_interval == 0:
        #     printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(all_iters, scheduler.get_lr()[0], loss.item()) + \
        #                 'Top-1 err = {:.6f},\t'.format(Top1_err / args.display_interval) + \
        #                 'Top-5 err = {:.6f},\t'.format(Top5_err / args.display_interval) + \
        #                 'data_time = {:.6f},\ttrain_time = {:.6f}'.format(data_time, (time.time() - t1) / args.display_interval)
        #     logging.info(printInfo)
        #     t1 = time.time()
        #     Top1_err, Top5_err = 0.0, 0.0
        #
        # if all_iters % args.save_interval == 0:
        #     save_checkpoint({
        #         'state_dict': model.state_dict(),
        #         }, all_iters)

    scheduler.step()
    printInfo = 'TRAIN epoch {}: lr = {:.6f},\tloss = {:.6f},\t'.format(epoch, scheduler.get_lr()[0], loss.item()) + \
                'Top-1 err = {:.6f},\t'.format(Top1_err / total) + \
                'Top-5 err = {:.6f},\t'.format(Top5_err / total) + \
                'data_time = {:.6f},\ttrain_time = {:.6f}'.format(data_time, (time.time() - t1) / total)
    logging.info(printInfo)
    t1 = time.time()
    Top1_err, Top5_err = 0.0, 0.0
    total = 0


    if all_iters % args.save_interval == 0:
        save_checkpoint({
            'state_dict': model.state_dict(),
            }, all_iters)


    print(architecture)
    return architecture, all_iters

def validate(model, device, args, *, all_iters=None):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    loss_function = args.loss_function
    val_dataprovider = args.val_dataprovider

    model.eval()
    max_val_iters = 250
    t1  = time.time()
    with torch.no_grad():
        for _ in range(1, max_val_iters + 1):
            data, target = val_dataprovider.next()
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)

            # 迭代出问题》》所以val出问题
            # output = model(data)
            # get_random_cand = lambda: tuple(np.random.randint(2) for i in range(7))  # 4
            # print(get_random_cand())
            # output = model(data, get_random_cand())
            architecture = [0, 0, 0, 0, 0, 0, 0]
            output = model(data, architecture) #

            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(all_iters, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(1 - top1.avg / 100) + \
              'Top-5 err = {:.6f},\t'.format(1 - top5.avg / 100) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    logging.info(logInfo)

def validate_one_epoch(architecture, model, device, args, *, all_iters=None):
    # objs = AvgrageMeter()
    # top1 = AvgrageMeter()
    # top5 = AvgrageMeter()

    total = 0
    Top1_err = 0
    Top5_err = 0
    train_loss = 0
    loss_function = args.loss_function
    val_dataprovider = args.val_dataprovider

    model.eval()
    # max_val_iters = 250
    max_val_iters = int(10000 // args.batch_size)
    t1  = time.time()
    with torch.no_grad():
        for _ in range(1, max_val_iters + 1):
            data, target = val_dataprovider.next()
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device)

            # 迭代出问题》》所以val出问题
            # output = model(data)
            # get_random_cand = lambda: tuple(np.random.randint(2) for i in range(7))  # 4
            # print(get_random_cand())
            # output = model(data, get_random_cand())
            # architecture = [0, 0, 0, 0, 0, 0, 0]
            # print(architecture)
            output = model(data, architecture) #

            loss = loss_function(output, target)
            train_loss += loss.item()
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            total += target.size(0)

            Top1_err += 1 - prec1.item() / 100
            Top5_err += 1 - prec5.item() / 100

            # n = data.size(0)
            # objs.update(loss.item(), n)
            # top1.update(prec1.item(), n)
            # top5.update(prec5.item(), n)


    printInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(all_iters,  train_loss / total) + \
                'Top-1 err = {:.6f},\t'.format(Top1_err / total) + \
                'Top-5 err = {:.6f},\t'.format(Top5_err / total)

    logging.info(printInfo)
    t1 = time.time()
    top1_err, top5_err = Top1_err / total, Top5_err / total
    Top1_err, Top5_err = 0.0, 0.0


    # logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(all_iters, objs.avg) + \
    #           'Top-1 err = {:.6f},\t'.format(Top1_err / total) + \
    #           'Top-5 err = {:.6f},\t'.format(Top5_err / total) + \
    #           'val_time = {:.6f}'.format(time.time() - t1)
    # logging.info(logInfo)
    return top1_err, top5_err

    all_iters = 0
    if args.auto_continue:
        lastest_model, iters = get_lastest_model()
        if lastest_model is not None:
            all_iters = iters
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('load from checkpoint')
            for i in range(iters):
                scheduler.step()

    if args.eval:
        if args.eval_resume is not None:

            # from network_origin import cifar_fast
            # model = cifar_fast(input_size=32, n_class=100)

            checkpoint = torch.load(args.eval_resume, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)

            """
            # model dict
            params = model.state_dict()
            for k, v in params.items():
                print(k)  # 只打印key值，不打印具体参数。
            print('**************************************')
            for k, v in checkpoint['state_dict'].items():
                print(k)

            print('**************************************')

            # 不匹配
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                # print(k)
                name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
                new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。
            model.load_state_dict(new_state_dict)
            """

            # model.load_state_dict(checkpoint, strict=True)
            validate(model, device, args, all_iters=all_iters)

        exit(0)
