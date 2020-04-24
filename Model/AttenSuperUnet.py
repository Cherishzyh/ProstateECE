import torch
import torch.nn as nn
from torch.autograd import Variable

from graphviz import Digraph


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, filters, stride=1):
        super(DoubleConv, self).__init__()
        self.conv1 = conv3x3(in_channels, filters, stride)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(filters, filters, stride)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int),
            nn.Sigmoid()
        )

        self.reasampler = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.relu = nn.ReLU(inplace=True)


    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        out = self.reasampler(x * psi)
        return out


class AttU_Net(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttU_Net, self).__init__()

        self.Conv1 = DoubleConv(in_channels, 64)
        self.Pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv2 = DoubleConv(64, 128)
        self.Pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv3 = DoubleConv(128, 256)
        self.Pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Conv4 = DoubleConv(256, 512)

        self.Up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.Att5 = Attention_block(F_g=256, F_l=512, F_int=512)
        self.Conv5 = DoubleConv(1024, 256)

        self.Up6 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.Att6 = Attention_block(F_g=128, F_l=256, F_int=256)
        self.Conv6 = DoubleConv(512, 128)

        self.Up7 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.Att7 = Attention_block(F_g=64, F_l=128, F_int=128)
        self.Conv7 = DoubleConv(256, 64)

        self.Conv_1x1 = nn.Conv2d(64, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path

        # 64*184*184
        conv_1 = self.Conv1(x)
        # 64*92*92
        pool_1 = self.Pool1(conv_1)
        # 128*92*92
        conv_2 = self.Conv2(pool_1)
        # 128*46*46
        pool_2 = self.Pool2(conv_2)
        # 256*46*46
        conv_3 = self.Conv3(pool_2)
        # 256*23*23
        pool_3 = self.Pool3(conv_3)
        # 512*23*23
        conv_4 = self.Conv4(pool_3)

        # 512*46*46
        up_5 = self.Up5(conv_4)
        # 256*46*46 + 512*46*46 = 512*46*46
        atten_5 = self.Att5(g=conv_3, x=up_5)
        # 512*46*46
        merge_5 = torch.cat((atten_5, up_5), dim=1)
        # 256*46*46
        conv_5 = self.Conv5(merge_5)
        # 256*92*92
        up_6 = self.Up6(conv_5)
        # 128*92*92 + 256*92*92
        atten_6 = self.Att6(g=conv_2, x=up_6)
        merge_6 = torch.cat((atten_6, up_6), dim=1)
        conv_6 = self.Conv6(merge_6)

        up_7 = self.Up7(conv_6)
        atten_7 = self.Att7(g=conv_1, x=up_7)
        merge_7 = torch.cat((atten_7, up_7), dim=1)
        conv_7 = self.Conv7(merge_7)

        d1 = self.Conv_1x1(conv_7)

        return d1


def make_dot(var, params=None):
    """
    画出 PyTorch 自动梯度图 autograd graph 的 Graphviz 表示.
    蓝色节点表示有梯度计算的变量Variables;
    橙色节点表示用于 torch.autograd.Function 中的 backward 的张量 Tensors.

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        assert all(isinstance(p, Variable) for p in params.values())
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled', shape='box', align='left',
                              fontsize='12', ranksep='0.1', height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()

    def size_to_str(size):
        return '(' + (', ').join(['%d' % v for v in size]) + ')'

    output_nodes = (var.grad_fn,) if not isinstance(var, tuple) else tuple(v.grad_fn for v in var)

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                # note: this used to show .saved_tensors in pytorch0.2, but stopped
                # working as it was moved to ATen and Variable-Tensor merged
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            elif var in output_nodes:
                dot.node(str(id(var)), str(type(var).__name__), fillcolor='darkolivegreen1')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)

    # 多输出场景 multiple outputs
    if isinstance(var, tuple):
        for v in var:
            add_nodes(v.grad_fn)
    else:
        add_nodes(var.grad_fn)
    return dot


def test():
    from tensorboardX import SummaryWriter
    writer = SummaryWriter('log')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = AttU_Net(in_channels=1, out_channels=1)
    # model = model.to(device)
    # vis_graph = make_dot(model, params=dict(model.named_parameters()))
    # vis_graph.view()
    dummy_input = torch.rand(20, 1, 184, 184)
    with SummaryWriter(comment='LeNet') as w:
        w.add_graph(model, (dummy_input))
    print(model)


if __name__ == '__main__':
    test()