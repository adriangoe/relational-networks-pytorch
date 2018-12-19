# -*- coding: utf-8 -*-
import click
import logging
import torch
from torchvision import transforms
import model
import util
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--mlp/--no-mlp', default=False)
@click.argument('n_epochs', type=int, default=20)
def main(data_path, mlp, n_epochs):
    '''Function to train and evaluate the model on the provided
    dataset.
    Args:
        - mlp: (bool) whether to use the MLP instead of the RN model
               (this is for comparisson purposes only)
        - n_epochs: (int) how many epochs to run
    '''
    logger = logging.getLogger(__name__)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('Detected device is %s' % device)

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(mode=None),
        transforms.Resize(75),
        transforms.ToTensor(),
    ])

    data_train = util.NotSoCLEVRDataset(csv_file=data_path + 'data_train.csv',
                                        img_file=data_path + 'data_train.h5',
                                        transform=img_transform)

    train_loader = DataLoader(data_train, batch_size=64,
                              shuffle=True, num_workers=1)
    logger.info('Loaded %s training datapoints' % len(data_train))

    data_test = util.NotSoCLEVRDataset(csv_file=data_path + 'data_test.csv',
                                       img_file=data_path + 'data_test.h5',
                                       transform=img_transform)

    test_loader = DataLoader(data_test, batch_size=64,
                             shuffle=False, num_workers=1)
    logger.info('Loaded %s test datapoints' % len(data_test))

    if mlp:
        model_ = model.CNN_MLP()
    else:
        model_ = model.RelationalNetwork()
    model_.to(device)
    logger.info('%s model loaded' % ('RelationalNetwork'
                                     if not mlp else 'MLP'))

    train_loss_history = []
    test_loss_history = []
    rel_acc_history = []
    non_rel_acc_history = []
    freq = 300

    for epoch in range(n_epochs):
        logger.info('Starting epoch %s out of %s' % (epoch, n_epochs))
        running_loss = 0.0

        for n_batch, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            tasks = torch.stack(batch['question']).t().float().to(device)
            target = batch['answer'].to(device)
            loss, preds = model_.train(images, tasks, target)

            running_loss += loss.item()

            if n_batch % freq == freq-1:
                train_loss_history.append(running_loss / freq)

                rel_correct = 0
                rel_total = 0.0
                non_rel_correct = 0
                non_rel_total = 0.0
                test_loss = 0.0
                with torch.no_grad():
                    # Get test set performance
                    for data in test_loader:
                        images = batch['image'].to(device)
                        tasks = torch.stack(batch['question']
                                            ).t().float().to(device)
                        target = batch['answer'].to(device)
                        types = batch['type']

                        loss, preds = model_.test(images, tasks, target)
                        c = (preds == target).squeeze()
                        test_loss += loss.item()

                        rel_correct += c[types == 1
                                         ].sum().cpu().data.tolist()
                        non_rel_correct += c[types == 0
                                             ].sum().cpu().data.tolist()
                        rel_total += c[types == 1].size()[0]
                        non_rel_total += c[types == 0].size()[0]

                # Keep track of performance metrics
                logger.info('[%d, %5d] train loss: %.3f, test loss: %.3f' %
                            (epoch + 1, n_batch + 1, running_loss / freq,
                             test_loss / len(test_loader)))
                logger.info('Test - relational: %s, non-relational: %s' %
                            (rel_correct / rel_total,
                             non_rel_correct / non_rel_total))
                rel_acc_history.append(rel_correct / rel_total)
                non_rel_acc_history.append(non_rel_correct / non_rel_total)

                test_loss_history.append(test_loss / len(test_loader))
                running_loss = 0.0

        plt.plot(train_loss_history)
        plt.plot(test_loss_history)
        plt.show()
        return


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
