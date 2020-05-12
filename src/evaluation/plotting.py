from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def plot_roc(prediction, target, class_names, path_to_save):
    """
    Code taken from a friend's repo.
    :param prediction:
    :param target:
    :param class_names:
    :param path_to_save:
    :return:
    """
    n_classes = 14
    false_positive_rate = dict()
    true_positive_rate = dict()
    roc_auc = dict()
    target = target.astype(int)

    # compute ROC
    for i in range(n_classes):
        false_positive_rate[i], true_positive_rate[i], _ = roc_curve(target[:, i], prediction[:, i])
        roc_auc[i] = auc(false_positive_rate[i], true_positive_rate[i])

    # Plot ROC curve
    plt.figure()
    auc_list = list()
    for i in range(n_classes):
        plt.plot(false_positive_rate[i], true_positive_rate[i], label=' {} '.format(class_names[i]))
        auc_list.append("class {0}, AUC = {1:0.2f}".format(class_names[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ResNet')
    plt.legend(loc="lower right")
    plt.savefig(path_to_save + "/roc_curve.png")

    # write AUC value
    with open(path_to_save + '/auc.txt', 'w') as f:
        for item in auc_list:
            f.write("%s\n" % item)

    print('In [plot_roc]: done')

