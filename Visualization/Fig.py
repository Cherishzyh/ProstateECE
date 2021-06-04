import os
import seaborn as sns

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def DmaxPPV(store_format='jpg', fontsize=12, s=100, is_save=False, store_path=r''):
    store_path = os.path.join(store_path, 'dmax_ppv')
    x_labels = ['D-max<1.5\n', '', 'D-max≥1.5\n']
    color_list1 = sns.color_palette('muted')

    fig, ax = plt.subplots()
    ax.set_title('PPV', fontsize=14)

    # PPV for Dmax
    ax.scatter(0.75, 50.0, c=color_list1[0], marker='*', s=s, label='PAGNet')
    ax.scatter(0.75, 21.4, c=color_list1[1], marker='o', s=s, label='Reader1')
    ax.scatter(0.75, 22.2, c=color_list1[2], marker='o', s=s, label='Reader2')
    ax.scatter(0.75, 44.4, c=color_list1[1], marker='^', s=s, label='Reader1+PAGNet')
    ax.scatter(0.75, 29.2, c=color_list1[2], marker='^', s=s, label='Reader2+PAGNet')
    ax.text(0.66, 50.0, str(50.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(0.84, 19.3, str(21.4) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(0.66, 22.2, str(22.2) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(0.66, 44.4, str(44.4) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(0.66, 29.2, str(29.2) + '%', fontsize=fontsize, color="black", ha='center', va="center")

    ax.scatter(1.25, 56.9, c=color_list1[0], marker='*', s=s)
    ax.scatter(1.25, 44.4, c=color_list1[2], marker='o', s=s)
    ax.scatter(1.25, 55.3, c=color_list1[1], marker='^', s=s)
    ax.scatter(1.25, 46.5, c=color_list1[1], marker='o', s=s)
    ax.scatter(1.25, 47.5, c=color_list1[2], marker='^', s=s)

    ax.text(1.16, 56.9, str(56.9) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.34, 46.3, str(46.5) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.16, 44.2, str(44.4) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.34, 54.5, str(55.3) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.16, 47.6, str(47.5) + '%', fontsize=fontsize, color="black", ha='center', va="center")

    ax.set_ylim(0, 70)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_yticklabels(['0', '10%', '20%', '30%', '40%', '50%', '60%'], fontsize=14)

    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([0.75, 1.0, 1.25])
    ax.set_xticklabels(x_labels, fontsize=14)

    ax.legend(loc='lower right', fontsize=14)

    plt.grid(True, 'major', 'both', ls='--', lw=.5, c='k', alpha=.2)

    fig.tight_layout()
    if is_save:
        plt.savefig(store_path+'.'+store_format, dpi=1200, format=store_format)
    else:
        plt.show()


def RISKPPV(store_format='jpg', fontsize=12, s=100, is_save=False, store_path=r''):
    store_path = os.path.join(store_path, 'risk_ppv')
    x_labels = ['Risk Group:\nLow', 'Risk Group:\nIntermediate', 'Risk Group:\nHigh']
    # [, 'PIRADS=3', 'PIRADS=4', 'PIRADS=5']
    color_list1 = sns.color_palette('muted')

    fig, ax = plt.subplots()
    ax.set_title('PPV', fontsize=14)

    ax.scatter(1.70, 20.0, c=color_list1[0], marker='*', s=s, label='PAGNet')
    ax.scatter(1.70, 25.0, c=color_list1[1], marker='o', s=s, label='Reader1')
    ax.scatter(1.705, 25.0, c=color_list1[2], marker='o', s=s, label='Reader2')
    ax.scatter(1.70, 33.3, c=color_list1[1], marker='^', s=s, label='Reader1+PAGNet')
    ax.scatter(1.70, 16.7, c=color_list1[2], marker='^', s=s, label='Reader2+PAGNet')
    ax.text(1.62, 20.0,  str(20.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.62, 25.0, str(25.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.785, 25.0, str(25.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.62, 33.3, str(33.3) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.62, 16.7, str(16.7) + '%', fontsize=fontsize, color="black", ha='center', va="center")

    ax.scatter(2.0, 50.0, c=color_list1[0], marker='*', s=s)
    ax.scatter(2.0, 26.1, c=color_list1[1], marker='o', s=s)
    ax.scatter(2.0, 22.7, c=color_list1[2], marker='o', s=s)
    ax.scatter(2.0, 36.4, c=color_list1[1], marker='^', s=s)
    ax.scatter(2.0, 29.6, c=color_list1[2], marker='^', s=s)
    ax.text(1.92, 50.0, str(50.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.92, 26.1, str(26.1) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.92, 22.7, str(22.7) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.92, 36.4, str(36.4) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.92, 29.6, str(29.6) + '%', fontsize=fontsize, color="black", ha='center', va="center")

    ax.scatter(2.30, 59.6, c=color_list1[0], marker='*', s=s)
    ax.scatter(2.30, 48.9, c=color_list1[1], marker='o', s=s)
    ax.scatter(2.30, 46.6, c=color_list1[2], marker='o', s=s)
    ax.scatter(2.30, 57.7, c=color_list1[1], marker='^', s=s)
    ax.scatter(2.30, 50.0, c=color_list1[2], marker='^', s=s)
    ax.text(2.39, 59.8, str(59.6) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.39, 48.7, str(48.9) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.22, 46.6, str(46.6) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.22, 56.8, str(57.7) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.22, 50.0, str(50.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")

    ax.set_ylim(0, 70)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_yticklabels(['0', '10%', '20%', '30%', '40%', '50%', '60%'], fontsize=14)

    ax.set_xlim(1.5, 2.5)
    ax.set_xticks([1.70, 2.0, 2.30])
    ax.set_xticklabels(x_labels, fontsize=14)

    # ax.set_title('Title')
    ax.legend(loc='lower right', fontsize=14)

    plt.grid(True, 'major', 'both', ls='--', lw=.5, c='k', alpha=.2)

    fig.tight_layout()
    if is_save:
        plt.savefig(store_path+'.'+store_format, dpi=1200, format=store_format)
    else:
        plt.show()


def PIRADSPPV(store_format='jpg', fontsize=12, s=100, is_save=False, store_path=r''):
    # PPV for PIRADS
    store_path = os.path.join(store_path, 'pirads_ppv')
    x_labels = ['PIRADS=3\n', 'PIRADS=4\n', 'PIRADS=5\n']
    color_list1 = sns.color_palette('muted')

    fig, ax = plt.subplots()
    ax.set_title('PPV', fontsize=fontsize)

    ax.scatter(2.70, 50.0, c=color_list1[0], marker='*', s=s, label='PAGNet')
    ax.scatter(2.70, 50.0, c=color_list1[1], marker='o', s=s, label='Reader1')
    ax.scatter(2.70, 0.0, c=color_list1[2], marker='o', s=s, label='Reader2')
    ax.scatter(2.70, 66.7, c=color_list1[1], marker='^', s=s, label='Reader1+PAGNet')
    ax.scatter(2.70, 20.0, c=color_list1[2], marker='^', s=s, label='Reader2+PAGNet')
    ax.scatter(2.70, 50.0, c=color_list1[0], marker='*', s=s)
    ax.text(2.62, 50.0, str(50.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.785, 50.0, str(50.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.62,  1.0, str( 0.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.62, 66.7, str(66.7) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.62, 20.0, str(20.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")

    ax.scatter(3.0, 60.0, c=color_list1[0], marker='*', s=s)
    ax.scatter(3.0, 20.0, c=color_list1[1], marker='o', s=s)
    ax.scatter(3.0, 33.3, c=color_list1[2], marker='o', s=s)
    ax.scatter(3.0, 44.4, c=color_list1[1], marker='^', s=s)
    ax.scatter(3.0, 35.0, c=color_list1[2], marker='^', s=s)
    ax.text(2.92, 60.0, str(60.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.92, 20.0, str(20.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(3.085, 32.5, str(33.3) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.92, 44.4, str(44.4) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.92, 35.3, str(35.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")

    ax.scatter(3.30, 57.1, c=color_list1[0], marker='*', s=s)
    ax.scatter(3.30, 47.3, c=color_list1[1], marker='o', s=s)
    ax.scatter(3.30, 45.5, c=color_list1[2], marker='o', s=s)
    ax.scatter(3.30, 54.8, c=color_list1[1], marker='^', s=s)
    ax.scatter(3.30, 49.0, c=color_list1[2], marker='^', s=s)
    ax.text(3.385, 57.1, str(57.1) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(3.385, 47.3, str(47.3) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(3.22, 45.5, str(45.5) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(3.22, 54.2, str(54.8) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(3.22, 49.0, str(49.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")

    ax.set_ylim(0, 70)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_yticklabels(['0', '10%', '20%', '30%', '40%', '50%', '60%'], fontsize=14)

    ax.set_xlim(2.5, 3.5)
    ax.set_xticks([2.70, 3.0, 3.30])
    ax.set_xticklabels(x_labels, fontsize=14)

    # ax.set_title('Title')
    ax.legend(loc='lower right', fontsize=14)

    plt.grid(True, 'major', 'both', ls='--', lw=.5, c='k', alpha=.2)

    fig.tight_layout()
    if is_save:
        plt.savefig(store_path+'.'+store_format, dpi=1200, format=store_format)
    else:
        plt.show()


def DmaxNPV(store_format='jpg', fontsize=12, s=100, is_save=False, store_path=r''):
    store_path = os.path.join(store_path, 'dmax_npv')
    x_labels = ['D-max<1.5\n', '', 'D-max≥1.5\n']
    color_list1 = sns.color_palette('muted')

    fig, ax = plt.subplots()
    ax.set_title('NPV', fontsize=14)

    # PPV for Dmax
    ax.scatter(0.75, 91.7, c=color_list1[0], marker='*', s=s, label='PAGNet')
    ax.scatter(0.75, 87.8, c=color_list1[1], marker='o', s=s, label='Reader1')
    ax.scatter(0.75, 88.5, c=color_list1[2], marker='o', s=s, label='Reader2')
    ax.scatter(0.75, 89.7, c=color_list1[1], marker='^', s=s, label='Reader1+PAGNet')
    ax.scatter(0.75, 91.7, c=color_list1[2], marker='^', s=s, label='Reader2+PAGNet')
    ax.scatter(0.75, 91.7, c=color_list1[0], marker='*', s=s)

    ax.text(0.66, 91.7, str(91.7) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(0.66, 87.8, str(87.8) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(0.845, 88.5, str(88.5) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(0.66, 89.7, str(89.7) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(0.845, 91.7, str(91.7) + '%', fontsize=fontsize, color="black", ha='center', va="center")

    ax.scatter(1.25, 72.7, c=color_list1[0], marker='*', s=s)
    ax.scatter(1.25, 76.8, c=color_list1[1], marker='o', s=s)
    ax.scatter(1.25, 68.4, c=color_list1[2], marker='o', s=s)
    ax.scatter(1.25, 77.8, c=color_list1[1], marker='^', s=s)
    ax.scatter(1.25, 78.6, c=color_list1[2], marker='^', s=s)
    ax.text(1.17, 72.7, str(72.7) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.17, 76.8, str(76.8) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.17, 68.4, str(68.4) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.335, 77.7, str(77.8) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.17, 78.8, str(78.6) + '%', fontsize=fontsize, color="black", ha='center', va="center")

    ax.set_ylim(65, 95)
    ax.set_yticks([65, 75, 85])
    ax.set_yticklabels(['65%', '75%', '85%'], fontsize=14)

    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([0.75, 1.0, 1.25])
    ax.set_xticklabels(x_labels, fontsize=14)

    ax.legend(loc='lower left', fontsize=14)

    plt.grid(True, 'major', 'both', ls='--', lw=.5, c='k', alpha=.2)

    fig.tight_layout()
    if is_save:
        plt.savefig(store_path+'.'+store_format, dpi=1200, format=store_format)
    else:
        plt.show()


def RISKNPV(store_format='jpg', fontsize=12, s=100, is_save=False, store_path=r''):
    store_path = os.path.join(store_path, 'risk_npv')
    x_labels = ['Risk Group:\nLow', 'Risk Group:\nIntermediate', 'Risk Group:\nHigh']
    # x_labels = ['Risk: Low', 'Risk: Intermediate', 'Risk: High']
    # [, 'PIRADS=3', 'PIRADS=4', 'PIRADS=5']
    color_list1 = sns.color_palette('muted')

    fig, ax = plt.subplots()
    ax.set_title('NPV', fontsize=14)

    ax.scatter(1.70, 94.4, c=color_list1[0], marker='*', label='PAGNet', s=s)
    ax.scatter(1.70, 94.7, c=color_list1[1], marker='o', label='Reader1', s=s)
    ax.scatter(1.705, 94.7, c=color_list1[2], marker='o', label='Reader2', s=s)
    ax.scatter(1.70, 95.0, c=color_list1[1], marker='^', label='Reader1+PAGNet', s=s)
    ax.scatter(1.70, 94.1, c=color_list1[2], marker='^', label='Reader2+PAGNet', s=s)

    ax.scatter(2.0, 88.1, c=color_list1[1], marker='o', s=s)
    ax.scatter(2.0, 86.8, c=color_list1[2], marker='o', s=s)
    ax.scatter(2.0, 87.3, c=color_list1[1], marker='^', s=s)
    ax.scatter(2.0, 90.5, c=color_list1[2], marker='^', s=s)
    ax.scatter(2.0, 87.8, c=color_list1[0], marker='*', s=s)
    ax.text(2.085, 88.2, str(88.1) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.92, 86.2, str(86.8) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.085, 86.5, str(87.3) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.92, 90.5, str(90.5) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(1.92, 87.9, str(87.8) + '%', fontsize=fontsize, color="black", ha='center', va="center")

    ax.scatter(2.30, 73.1, c=color_list1[1], marker='o', s=s)
    ax.scatter(2.30, 65.7, c=color_list1[2], marker='o', s=s)
    ax.scatter(2.30, 76.8, c=color_list1[1], marker='^', s=s)
    ax.scatter(2.30, 77.1, c=color_list1[2], marker='^', s=s)
    ax.scatter(2.30, 72.3, c=color_list1[0], marker='*', s=s)
    ax.text(2.385, 73.3, str(73.1) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.22, 65.7, str(65.7) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.22, 76.6, str(76.8) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.385, 77.1, str(77.1) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.22, 72.3, str(72.3) + '%', fontsize=fontsize, color="black", ha='center', va="center")

    axins2 = ax.inset_axes([0.02, 0.44, 0.32, 0.34])
    axins2.set_ylim(93.0, 96.0)
    axins2.set_xlim(1.65, 1.75)
    axins2.set_yticks([])
    axins2.set_xticks([])

    axins = ax.inset_axes([0.02, 0.44, 0.32, 0.34])
    axins.scatter(1.70, 94.4, c=color_list1[0], marker='*' , s=s)
    axins.scatter(1.70, 94.7, c=color_list1[1], marker='o' , s=s)
    axins.scatter(1.705, 94.7, c=color_list1[2], marker='o', s=s)
    axins.scatter(1.70, 95.0, c=color_list1[1], marker='^' , s=s)
    axins.scatter(1.70, 94.1, c=color_list1[2], marker='^' , s=s)
    # axins.text(1.675, 94.4, str(94.4) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    # axins.text(1.73, 94.7, str(94.7) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    # axins.text(1.675, 94.7, str(94.7) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    # axins.text(1.675, 95.0, str(95.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    # axins.text(1.675, 94.1, str(94.1) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    axins.text(1.68, 94.4, str(94.4) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    axins.text(1.722, 94.7, str(94.7) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    axins.text(1.68, 94.7, str(94.7) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    axins.text(1.68, 95.0, str(95.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    axins.text(1.68, 94.1, str(94.1) + '%', fontsize=fontsize, color="black", ha='center', va="center")

    axins.set_ylim(94.0, 95.1)
    axins.set_yticks([])
    axins.set_xticks([])

    mark_inset(ax, axins2, loc1=2, loc2=1, fc="none", ec='k', lw=1)

    ax.set_ylim(65, 100)
    ax.set_yticks([65, 75, 85, 95])
    ax.set_yticklabels(['65%', '75%', '85%', '95%'], fontsize=14)

    ax.set_xlim(1.5, 2.5)
    ax.set_xticks([1.70, 2.0, 2.30])
    ax.set_xticklabels(x_labels, fontsize=14)

    # ax.set_title('Title')
    ax.legend(loc='lower left', fontsize=14)

    plt.grid(True, 'major', 'both', ls='--', lw=.5, c='k', alpha=.2)
    fig.tight_layout()
    if is_save:
        plt.savefig(store_path+'.'+store_format, dpi=1200, format=store_format)
    else:
        plt.show()


def PIRADSNPV(store_format='jpg', fontsize=12, s=100, is_save=False, store_path=r''):
    # PPV for PIRADS
    store_path = os.path.join(store_path, 'pirads_npv')
    x_labels = ['PIRADS=3\n', 'PIRADS=4\n', 'PIRADS=5\n']
    color_list1 = sns.color_palette('muted')

    fig, ax = plt.subplots()
    ax.set_title('NPV', fontsize=14)

    ax.scatter(2.70, 93.3, c=color_list1[0], marker='*', s=s, label='PAGNet')
    ax.scatter(2.70, 100.0, c=color_list1[1], marker='o', s=s, label='Reader1')
    ax.scatter(2.70, 89.3, c=color_list1[2], marker='o', s=s,  label='Reader2')
    ax.scatter(2.70, 96.6, c=color_list1[1], marker='^', s=s, label='Reader1+PAGNet')
    ax.scatter(2.70, 92.6, c=color_list1[2], marker='^', s=s, label='Reader2+PAGNet')
    ax.scatter(2.70, 93.3, c=color_list1[0], marker='*', s=s)
    ax.text(2.61, 93.3, str(93.30) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.795, 100.0, str(100.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.61, 89.3, str(89.30) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.61, 96.6, str(96.60) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.795, 92.6, str(92.60) + '%', fontsize=fontsize, color="black", ha='center', va="center")

    ax.scatter(3.0, 78.3, c=color_list1[1], marker='o', s=s)
    ax.scatter(3.0, 82.6, c=color_list1[2], marker='o', s=s)
    ax.scatter(3.0, 82.7, c=color_list1[1], marker='^', s=s)
    ax.scatter(3.0, 85.4, c=color_list1[2], marker='^', s=s)
    ax.scatter(3.0, 86.3, c=color_list1[0], marker='*', s=s)
    ax.text(3.10, 86.3, str(86.3) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(3.10, 78.3, str(78.3) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.91, 82.6, str(82.6) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(3.10, 82.7, str(82.7) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(2.91, 84.9, str(85.4) + '%', fontsize=fontsize, color="black", ha='center', va="center")

    ax.scatter(3.30, 69.8, c=color_list1[1], marker='o', s=s)
    ax.scatter(3.30, 62.7, c=color_list1[2], marker='o', s=s)
    ax.scatter(3.30, 73.0, c=color_list1[1], marker='^', s=s)
    ax.scatter(3.30, 75.0, c=color_list1[2], marker='^', s=s)
    ax.scatter(3.30, 68.8, c=color_list1[0], marker='*', s=s)
    ax.text(3.21, 68.8, str(68.8) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(3.395, 69.8, str(69.8) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(3.21, 62.7, str(62.7) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(3.21, 73.0, str(73.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")
    ax.text(3.395, 75.0, str(75.0) + '%', fontsize=fontsize, color="black", ha='center', va="center")

    ax.set_ylim(60, 105)
    ax.set_yticks([60, 70, 80, 90, 100])
    ax.set_yticklabels(['60%', '70%', '80%', '90%', '100%'], fontsize=14)

    ax.set_xlim(2.5, 3.5)
    ax.set_xticks([2.70, 3.0, 3.30])
    ax.set_xticklabels(x_labels, fontsize=14)

    # ax.set_title('Title')
    ax.legend(loc='lower left', fontsize=14)

    plt.grid(True, 'major', 'both', ls='--', lw=.5, c='k', alpha=.2)

    fig.tight_layout()
    if is_save:
        plt.savefig(store_path+'.'+store_format, dpi=1200, format=store_format)
    else:
        plt.show()


if __name__ == '__main__':
    fontsize = 17
    store_path = r'C:\Users\ZhangYihong\Desktop\test'
    is_save = True
    store_format='tif'
    DmaxPPV(fontsize=fontsize, s=120, is_save=is_save, store_format=store_format, store_path=store_path)
    RISKPPV(fontsize=fontsize, s=120, is_save=is_save, store_format=store_format, store_path=store_path)
    PIRADSPPV(fontsize=fontsize, s=120, is_save=is_save, store_format=store_format, store_path=store_path)
    DmaxNPV(fontsize=fontsize, s=120, is_save=is_save, store_format=store_format, store_path=store_path)
    RISKNPV(fontsize=fontsize, s=120, is_save=is_save, store_format=store_format, store_path=store_path)
    PIRADSNPV(fontsize=fontsize, s=120, is_save=is_save, store_format=store_format, store_path=store_path)
