import torch
import torch.nn as nn
from blocks import Shufflenet, Shuffle_Xception #, SeModule, Inverted_Bottleneck


class ShuffleNetV2_OneShot(nn.Module):

    def __init__(self, input_size=224, n_class=1000):
        super(ShuffleNetV2_OneShot, self).__init__()

        assert input_size % 32 == 0

        self.stage_repeats = [4, 4, 8, 4]
        self.stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel, affine=False),
            nn.ReLU(inplace=True),
        )

        self.features = torch.nn.ModuleList()
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                base_mid_channels = outp // 2
                mid_channels = int(base_mid_channels)
                archIndex += 1
                self.features.append(torch.nn.ModuleList())
                for blockIndex in range(4):
                    if blockIndex == 0:
                        # print('Shuffle3x3')
                        self.features[-1].append(
                            Shufflenet(inp, outp, mid_channels=mid_channels, ksize=3, stride=stride))
                    elif blockIndex == 1:
                        # print('Shuffle5x5')
                        self.features[-1].append(
                            Shufflenet(inp, outp, mid_channels=mid_channels, ksize=5, stride=stride))
                    elif blockIndex == 2:
                        # print('Shuffle7x7')
                        self.features[-1].append(
                            Shufflenet(inp, outp, mid_channels=mid_channels, ksize=7, stride=stride))
                    elif blockIndex == 3:
                        # print('Xception')
                        self.features[-1].append(
                            Shuffle_Xception(inp, outp, mid_channels=mid_channels, stride=stride))
                    else:
                        raise NotImplementedError
                input_channel = output_channel

        self.archLen = archIndex
        # self.features = nn.Sequential(*self.features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(
                input_channel, self.stage_out_channels[
                    -1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1], affine=False),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    def forward(self, x, architecture=False):

        # architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
        assert self.archLen == len(architecture)

        x = self.first_conv(x)

        for archs, arch_id in zip(self.features, architecture):
            x = archs[arch_id](x)

        x = self.conv_last(x)

        x = self.globalpool(x)

        x = self.dropout(x)
        x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# cifar_fast
class cifar_fast(nn.Module):
    def __init__(self, input_size=224, n_class=1000):
        super(cifar_fast, self).__init__()
        assert input_size % 32 == 0
        #
        # self.stage_repeats = [4, 4, 8, 4]
        # self.stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]

        # depth
        self.stage_repeats = [3, 1, 3] # 1+3+1+3= 8layers
        self.stage_out_channels = [-1, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(2)

        # building first layer
        input_channel = self.stage_out_channels[1] # 64

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(input_channel, affine=False),
            nn.ReLU(inplace=True),
        )

        self.features = torch.nn.ModuleList()
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)): # 0, 1, 2
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2] # 128, 256, 512
            for i in range(numrepeat): # 0,1,2
                if i == 0: # 通过pooling downsample
                    inp, outp, stride = input_channel, output_channel, 1
                else:
                    inp, outp, stride = output_channel, output_channel, 1

                base_mid_channels = outp // 2
                mid_channels = int(base_mid_channels)
                archIndex += 1
                self.features.append(torch.nn.ModuleList())
                for blockIndex in range(4):
                    """
                    # shuffle Unit 由多个conv组成
                    # 0, 1, 2, 3 四个

                    if blockIndex == 0:
                        # print('Shuffle3x3')
                        self.features[-1].append(
                            Shufflenet(inp, outp, mid_channels=mid_channels, ksize=3, stride=stride))
                    elif blockIndex == 1:
                        # print('Shuffle5x5')
                        self.features[-1].append(
                            Shufflenet(inp, outp, mid_channels=mid_channels, ksize=5, stride=stride))
                    elif blockIndex == 2:
                        # print('Shuffle7x7')
                        self.features[-1].append(
                            Shufflenet(inp, outp, mid_channels=mid_channels, ksize=7, stride=stride))
                    elif blockIndex == 3:
                        # print('Xception')
                        self.features[-1].append(
                            Shuffle_Xception(inp, outp, mid_channels=mid_channels, stride=stride))
                    """
                    # 普通conv 一个conv组成
                    if blockIndex == 0:
                        self.features[-1].append(nn.Sequential(
                            nn.Conv2d(inp, outp, kernel_size=3, stride=stride, padding=1, bias=False),
                            nn.BatchNorm2d(outp),
                            nn.ReLU(inplace=True),
                            # nn.MaxPool2d(2)
                        ))

                    elif blockIndex == 1:
                        self.features[-1].append(nn.Sequential(
                            nn.Conv2d(inp, outp, kernel_size=5, stride=stride, padding=2, bias=False),
                            nn.BatchNorm2d(outp),
                            nn.ReLU(inplace=True),
                            # nn.MaxPool2d(2)
                        ))

                    # dwconv
                    elif blockIndex == 2:
                        # print('dwconv3x3')
                        # self.features[-1].append(
                        #     self.dwconv(inp, outp, ksize=3, stride=stride))

                        # self.features[-1]= self.dwconv(inp, outp, ksize=3, stride=stride)

                        in_channels, out_channels, ksize = inp, outp, 3
                        self.features[-1].append(nn.Sequential(
                            # dw
                            nn.Conv2d(in_channels, in_channels, ksize, stride, ksize // 2, groups=in_channels,
                                      bias=False),
                            nn.BatchNorm2d(in_channels),
                            nn.ReLU(inplace=True),
                            # pw
                            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True))
                        )

                    elif blockIndex == 3:
                        # print('dwconv5x5')
                        # self.features[-1].append(
                        #     self.dwconv(inp, outp, ksize=5, stride=stride))
                        # self.features[-1] = self.dwconv(inp, outp, ksize=5, stride=stride)
                        in_channels, out_channels, ksize = inp, outp, 5
                        self.features[-1].append(nn.Sequential(
                            # dw
                            nn.Conv2d(in_channels, in_channels, ksize, stride, ksize // 2, groups=in_channels,
                                      bias=False),
                            nn.BatchNorm2d(in_channels),
                            nn.ReLU(inplace=True),
                            # pw
                            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True))
                        )

                    # Inverted_Bottleneck
                    # self.bneck = nn.Sequential(
                    #     Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
                    #     Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
                    #     Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
                    #     Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
                    #     Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
                    #     Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
                    #     Block(3, 40, 240, 80, hswish(), None, 2),
                    #     Block(3, 80, 200, 80, hswish(), None, 1),
                    #     Block(3, 80, 184, 80, hswish(), None, 1),
                    #     Block(3, 80, 184, 80, hswish(), None, 1),
                    #     Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
                    #     Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
                    #     Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
                    #     Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
                    #     Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
                    # )

                    elif blockIndex == 4:
                        # print('Inverted_Bottleneck')
                        self.features[-1].append(
                            Inverted_Bottleneck(3, inp, outp, mid_channels, nn.ReLU(inplace=True), None, 1))


                    elif blockIndex == 5:
                        # print('Inverted_Bottleneck5*5')
                        self.features[-1].append(
                            Inverted_Bottleneck(5, inp, outp, mid_channels, nn.ReLU(inplace=True), None, 1))

                    else:
                        raise NotImplementedError
                input_channel = output_channel

        self.archLen = archIndex
        # self.features = nn.Sequential(*self.features)

        """
        self.conv_last = nn.Sequential(
            nn.Conv2d(
                input_channel, self.stage_out_channels[
                    -1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1], affine=False),
            nn.ReLU(inplace=True),
        )
        self.globalpool = nn.AvgPool2d(7)
        """

        self.globalpool = nn.AvgPool2d(4)
        # self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(self.stage_out_channels[-1], n_class, bias=False))
        self._initialize_weights()

    # def forward(self, x, architecture=[0, 0, 0, 0, 0, 0, 0]):
    def forward(self, x, architecture):
        # architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]

        assert self.archLen == len(architecture)

        x = self.first_conv(x)

        index=0
        for archs, arch_id in zip(self.features, architecture):
            index = index+1
            x = archs[arch_id](x)

            # arch_id 第一个和第五个会有跳层
            if index == 1:
                x = self.pool(x) #
                residual = x
            elif index == 3:
                x = x + residual
            elif index == 4:
                x = self.pool(x) #
            elif index == 5:
                x = self.pool(x) #
                residual = x
            elif index == 7:
                x = x + residual

        # x = self.conv_last(x)

        x = self.globalpool(x)

        # x = self.dropout(x)
        # x = self.dropout(x)

        # x = x.contiguous().view(-1, self.stage_out_channels[-1])
        x = x.view(-1, self.stage_out_channels[-1])
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    architecture = [0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2]
    # scale_list = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    # scale_ids = [6, 5, 3, 5, 2, 6, 3, 4, 2, 5, 7, 5, 4, 6, 7, 4, 4, 5, 4, 3]
    # channels_scales = []
    # for i in range(len(scale_ids)):
    #     channels_scales.append(scale_list[scale_ids[i]])

    # architecture = [4, 4, 4, 4, 4, 4, 4]
    architecture = [0, 0, 0, 0, 0, 0, 0]

    # ------
    import torch
    from torchvision import models
    from torchsummary import summary
    from ptflops import get_model_complexity_info

    input_size = 32  # 224 224 #
    # inputs = torch.randn(3, 32, 32).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # model = ShuffleNetV2_OneShot(n_class=100)
    model = cifar_fast(input_size=input_size, n_class=100)
    print(model)

    test_data = torch.rand(5, 3, input_size, input_size)
    test_outputs = model(test_data, architecture)
    print(test_outputs.size())

    model = model.to(device)

    # part1
    with  torch.cuda.device(0):
        # 把choice拿到forward里
        # choice = {
        #     0: {'conv_0': [0], 'conv_1': [0], 'conv_2': [0], 'rate': 0},
        #     2: {'conv_0': [0], 'conv_1': [0], 'conv_2': [0], 'rate': 0}}

        # flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=True)
        flops, params = get_model_complexity_info(model, (3, input_size, input_size), as_strings=True, print_per_layer_stat=True)
        # part2
        summary(model, (3, input_size, input_size))
        print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))