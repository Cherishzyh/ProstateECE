import matplotlib.pyplot as plt
import numpy as np

def DrawPaperImage():
    fig = plt.figure(figsize=(12, 10))
    l = 0.92
    b = 0.12
    w = 0.015
    h = 0.37
    rect = [l, b, w, h]
    cbar_ax = fig.add_axes(rect)

    plt.subplot(121)
    plt.axis('off')
    plt.imshow(np.squeeze(t2), cmap='gray')
    plt.subplot(122)
    plt.axis('off')
    plt.imshow(np.squeeze(merged_image), cmap='jet')
    plt.colorbar(cax=cbar_ax)

    plt.gca().set_axis_off()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.subplots_adjust(left=None, bottom=None, right=0.9, top=None,
                        wspace=0.01, hspace=0.01)

    # plt.savefig(r'/home/zhangyihong/Documents/ProstateECE/Paper/' + str(i) + '.tif', format='tif', dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(r'/home/zhangyihong/Documents/ProstateECE/Paper/' + str(i) + '.eps', format='eps', dpi=600, bbox_inches='tight', pad_inches=0.05)
    # plt.show()
    plt.close()
    plt.clf()

    plt.show()