import torch
from torch.utils.data import DataLoader

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from SSHProject.CnnTools.T4T.Utility.Data import *

from SYECE.model import ResNeXt
# from SYECE.ModelWithoutDis import ResNeXt


def ModelJSPH(data_type):

    data_root = r'/home/zhangyihong/Documents/ProstateECE/NPYNoDivide'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (192, 192)
    batch_size = 2
    model_root = r'/home/zhangyihong/Documents/ProstateECE/Model/ResNeXt_CBAM_CV_20200820/CV_0/31--5.778387.pt'
    # model_root = r'/home/zhangyihong/Documents/ProstateECE/Model/ResNeXt_CBAM_CV_20200814/CV_1/154--7.698224.pt'

    spliter = DataSpliter()
    sub_list = spliter.LoadName(data_root + '/{}-name.csv'.format(data_type))

    if data_type == 'test':
        data = DataManager(sub_list=sub_list)
        data.AddOne(Image2D(data_root + '/T2Slice/Test', shape=input_shape))
        data.AddOne(Image2D(data_root + '/AdcSlice/Test', shape=input_shape))
        data.AddOne(Image2D(data_root + '/DwiSlice/Test', shape=input_shape))
        # data.AddOne(Image2D(data_root + '/DistanceMap/Test', shape=input_shape, is_roi=True))
        # data.AddOne(Label(data_root + '/label.csv', label_tag='Negative'), is_input=False)
        data.AddOne(Label(data_root + '/label.csv', label_tag='Positive'), is_input=False)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)
    else:
        data = DataManager(sub_list=sub_list)
        data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
        data.AddOne(Image2D(data_root + '/AdcSlice', shape=input_shape))
        data.AddOne(Image2D(data_root + '/DwiSlice/', shape=input_shape))
        # data.AddOne(Image2D(data_root + '/DistanceMap', shape=input_shape, is_roi=True))
        # data.AddOne(Label(data_root + '/label.csv', label_tag='Negative'), is_input=False)
        data.AddOne(Label(data_root + '/label.csv', label_tag='Positive'), is_input=False)
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    model = ResNeXt(3, 2).to(device)
    model.load_state_dict(torch.load(model_root))
    fc_out_list = []
    model.eval()
    for inputs, outputs in data_loader:
        inputs = MoveTensorsToDevice(inputs, device)
        model_pred = model(*inputs)
        fc_out_list.extend(model_pred[1].cpu().data.numpy().squeeze().tolist())

    fcn = np.array(fc_out_list)
    np.save(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/best_fcn_ResNeXt_test.npy', fcn)


def ModelSUH():
    data_root = r'/home/zhangyihong/Documents/ProstateECE/SUH_Dwi1500'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (192, 192)
    batch_size = 2
    # model_root = r'/home/zhangyihong/Documents/ProstateECE/Model/ResNeXt_CBAM_CV_20200820/CV_0/31--5.778387.pt'
    model_root = r'/home/zhangyihong/Documents/ProstateECE/Model/ResNeXt_CBAM_CV_20200814/CV_1/154--7.698224.pt'

    data = DataManager()
    data.AddOne(Image2D(data_root + '/T2Slice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/AdcSlice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/DwiSlice', shape=input_shape))
    data.AddOne(Image2D(data_root + '/DistanceMap', shape=input_shape, is_roi=True))
    data.AddOne(Label(data_root + '/label_negative.csv', label_tag='Negative'), is_input=False)
    # data.AddOne(Label(data_root + '/label_negative.csv', label_tag='Positive'), is_input=False)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    model = ResNeXt(3, 2).to(device)
    model.load_state_dict(torch.load(model_root))

    fc_out_list = []
    model.eval()
    for inputs, outputs in data_loader:
        inputs = MoveTensorsToDevice(inputs, device)
        model_pred = model(*inputs)

        if isinstance((1 - model_pred[0][:, 1]).cpu().data.numpy().squeeze().tolist(), float):
                fc_out_list.append(model_pred[1].cpu().data.numpy().squeeze().tolist())
        else:
                fc_out_list.extend(model_pred[1].cpu().data.numpy().squeeze().tolist())

    fcn = np.array(fc_out_list)
    np.save(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/best_fcn_PAGNet_suh.npy', fcn)



def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        if s == 0:
            p1 = plt.scatter(x, y, c='b')
        elif s == 1:
            p2 = plt.scatter(x, y, c='r')
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.legend([p1, p2], ["Negative", "Positive"])
    plt.show()


def TSNE_plot():
    perplexity = 30
    # TRAIN
    fcn_out = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/aver_fcn_PAGNet_train.npy')
    tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=5000)
    x_embedded = tsne.fit_transform(fcn_out)
    labels = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/JSPH_TRAIN_label.npy')
    plot_with_labels(x_embedded, labels)

    # TEST
    fcn_out = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/aver_fcn_PAGNet_test.npy')
    tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=5000)
    x_embedded = tsne.fit_transform(fcn_out)
    labels = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/JSPH_label.npy')
    plot_with_labels(x_embedded, labels)

    # SUH
    fcn_out = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/aver_fcn_PAGNet_suh.npy')
    tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=1000)
    x_embedded = tsne.fit_transform(fcn_out)
    labels = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/SUH_label.npy')
    plot_with_labels(x_embedded, labels)

    # TRAIN
    fcn_out = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/aver_fcn_ResNeXt_train.npy')
    tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=5000)
    x_embedded = tsne.fit_transform(fcn_out)
    labels = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/JSPH_TRAIN_label.npy')
    plot_with_labels(x_embedded, labels)

    # TEST
    fcn_out = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/aver_fcn_ResNeXt_test.npy')
    tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=5000)
    x_embedded = tsne.fit_transform(fcn_out)
    labels = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/JSPH_label.npy')
    plot_with_labels(x_embedded, labels)

    # SHU
    fcn_out = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/aver_fcn_ResNeXt_suh.npy')
    tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=5000)
    x_embedded = tsne.fit_transform(fcn_out)
    labels = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/SUH_label.npy')
    plot_with_labels(x_embedded, labels)


def TSNE_plot_csv():
    import pandas as pd
    perplexity = 30
    # TRAIN
    fcn_out = pd.read_csv(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/pagnet_train.csv')
    label_list = []
    feature_list = []
    for index in fcn_out.index:
        label_list.append(fcn_out.loc[index]['label'])
        feature_list.append(fcn_out.iloc[index][2:].tolist())

    tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=5000)
    x_embedded = tsne.fit_transform(np.array(feature_list))

    plot_with_labels(x_embedded, np.array(label_list))


def WriteCSV():
    label_train = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/JSPH_TRAIN_label.npy').tolist()
    label_test = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/JSPH_label.npy').tolist()
    label_suh = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/SUH_label.npy').tolist()

    pagnet_train = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/aver_fcn_PAGNet_train.npy')
    pagnet_test = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/aver_fcn_PAGNet_test.npy')
    pagnet_suh = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/aver_fcn_PAGNet_suh.npy')

    resnext_train = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/aver_fcn_ResNeXt_train.npy')
    resnext_test = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/aver_fcn_ResNeXt_test.npy')
    resnext_suh = np.load(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/aver_fcn_ResNeXt_suh.npy')

    dict = {'label': label_train}
    for index in range(pagnet_train.shape[-1]):
        dict[index] = pagnet_train[:, index]
    df = pd.DataFrame(dict)
    df.to_csv(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/pagnet_train.csv')

    dict = {'label': label_train}
    for index in range(resnext_train.shape[-1]):
        dict[index] = resnext_train[:, index]
    df = pd.DataFrame(dict)
    df.to_csv(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/resnext_train.csv')

    dict = {'label': label_test}
    for index in range(pagnet_test.shape[-1]):
        dict[index] = pagnet_test[:, index]
    df = pd.DataFrame(dict)
    df.to_csv(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/pagnet_test.csv')

    dict = {'label': label_test}
    for index in range(resnext_test.shape[-1]):
        dict[index] = resnext_test[:, index]
    df = pd.DataFrame(dict)
    df.to_csv(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/resnext_test.csv')

    dict = {'label': label_suh}
    for index in range(pagnet_suh.shape[-1]):
        dict[index] = pagnet_suh[:, index]
    df = pd.DataFrame(dict)
    df.to_csv(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/pagnet_suh.csv')

    dict = {'label': label_suh}
    for index in range(resnext_suh.shape[-1]):
        dict[index] = resnext_suh[:, index]
    df = pd.DataFrame(dict)
    df.to_csv(r'/home/zhangyihong/Documents/ProstateECE/Result/NPY/FcnOut/resnext_suh.csv')


if __name__ == '__main__':
    # TSNE_plot()
    TSNE_plot_csv()




