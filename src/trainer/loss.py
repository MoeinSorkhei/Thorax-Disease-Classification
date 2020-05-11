import torch


from globals import device
# this device is accessible in all the functions in this file
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def compute_wcel(fx, labels):
    """
    Weighted Cross Entropy Loss for multi-label classification, Eq. (1) in the paper.
    :param fx: the output (prediction) of the unified network, tensor of shape (B, 14).
    :param labels: the true labels, tensor of shape (B, 14).
    :return average WCEL.
    """
    # converting the labels batch to from Long tensor to Float tensor (otherwise won't work on GPU)
    labels = labels.float()

    p = labels.sum()  # number of 1's in a batch
    n = labels.shape[0] * labels.shape[1] - p  # number of 0's in a batch
    beta_p = (p + n) / (p + 1e-5)  # avoid zero in the denominator (also for the log function)
    beta_n = (p + n) / (n + 1e-5)

    absences = 1 - labels  # this would have 1's where the labels is 0 (absence of a disease)
    loss = (-beta_p * torch.log(fx + 1e-6) * labels).sum() + \
           (-beta_n * torch.log(1 - fx + 1e-6) * absences).sum()

    batch_size = fx.shape[0]
    loss_avg = loss / batch_size
    return loss_avg


def compute_val_loss(network, val_loader):
    """
    Computes the validation loss and return its float number.
    :param network: the trained net.
    :param val_loader: validation data loader
    :return: the float number representing the validation loss
    """
    print(f'In [compute_val_loss]: computing validation loss for {len(val_loader)} batches...')
    val_loss = 0
    for i_batch, batch in enumerate(val_loader):
        img_batch, label_batch = batch['image'].to(device), batch['label'].to(device)
        val_pred = network(img_batch)

        batch_loss = compute_wcel(fx=val_pred, labels=label_batch)
        val_loss += batch_loss.item()

    val_avg = val_loss / len(val_loader)  # taking the average over all the batches
    val_avg = round(val_avg, 3)  # round to three floating points

    print(f'In [compute_val_loss]: validation loss: {val_avg} \n')
    return val_avg
