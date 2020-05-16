from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        auc_list.append("class {0}, AUC = {1:0.4f}".format(class_names[i], roc_auc[i]))

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


def plot_bbox(img, predicted=None, ground_truth=None, heatmap=None):
    """
    Plots the predicted and/or real bounding box on top of the x-ray image.
    If heatmap is given, also plot it in a separate subplot.
    
    :param img: the x-ray Image to use as background.
    :param predicted: an array of predicted BBoxes. (green)
    :param ground_truth: the real BBox. (blue)
    :param heatmap: heatmap that was used to generate predicted BBox.
    """
    # If no heatmap, just create a single subplot.
    if heatmap is None:
        fig, ax1 = plt.subplots(1, 1);

    # If given, plot heatmap in second subplot.
    else :
        fig, (ax1, ax2) = plt.subplots(1, 2);
        extent = (0, 1, 0, 1)
        ax2.imshow(img, extent=extent, cmap='gray')
        ax2.imshow(heatmap, extent=extent, cmap='jet', alpha=0.6)
        ax2.axis('off')

    # Plot x-ray image beneath bboxes.
    ax1.imshow(img, cmap='gray')
    ax1.axis('off')

    # Ordered so ground truth bbox is drawn below predicted.
    bboxes = [ground_truth] + predicted
    colors = ['b'] + ['g'] * len(predicted)

    # Plot all given bboxes.
    for i, box in enumerate(bboxes):
        if box is not None:
            ax1.add_patch(patches.Rectangle(
                (box.x, box.y), box.width, box.height,
                fill=False,
                edgecolor=colors[i]
            ))

    plt.tight_layout()
    plt.show()