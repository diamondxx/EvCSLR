import torch


def masked_bn(inputs, len_x):
    def pad(tensor, length):
        return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])
    x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
    print(x.shape)
    # x = self.conv2d(x)
    x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                   for idx, lgt in enumerate(len_x)])
    print(x.shape)
    return x
x = torch.randn(120,3,224,224)
len_x = (40,60)
masked_bn(x, len_x)