from tensorboardX import SummaryWriter


def VisNet():
    with SummaryWriter(log_dir='logs', comment='Net') as w:
        w.add_graph(model, (input, ))


def VisLoss():
    writer = SummaryWriter(log_dir='logs', comment='Loss')
