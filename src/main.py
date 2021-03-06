# -*- coding: utf-8 -*-
import click
import logging
import torch
from torchvision import transforms
from src import model
from src import util
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import time


@click.command()
@click.argument('data_path', type=click.Path(exists=True))
@click.option('--mlp/--no-mlp', default=False)
@click.option('--lstm/--no-lstm', default=False)
@click.argument('n_epochs', type=int, default=20)
def main(data_path, mlp, lstm, n_epochs):
    '''Function to train and evaluate the model on the provided
    dataset.
    Args:
        - mlp: (bool) whether to use the MLP instead of the RN model
               (this is for comparisson purposes only)
        - lstm: (bool) whether to use an LSTM on raw sentences or the
                prepared encodings
        - n_epochs: (int) how many epochs to run
    '''
    name = time.strftime("%Y%m%d-%H%M%S") + ('_mlp' if mlp else '_rn')
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

    if lstm:
        # Count vocabulary and prepare word encodings
        text_util = util.TextUtil(data_path + 'data_train.csv', 'question')
        logger.info('Prepared LSTM vocabulary with %s words'
                    % text_util.vocab_size)
        vocab = text_util.vocab_size
    else:
        vocab = None

    if mlp:
        model_ = model.CNN_MLP(lstm=lstm, vocab=vocab)
    else:
        model_ = model.RelationalNetwork(lstm=lstm, vocab=vocab)
    model_.to(device)
    logger.info('%s model loaded' % ('RelationalNetwork'
                                     if not mlp else 'MLP'))

    train_loss_history = []
    test_loss_history = []
    rel_acc_history = []
    non_rel_acc_history = []
    freq = 500

    for epoch in range(n_epochs):
        logger.info('Starting epoch %s out of %s' % (epoch + 1, n_epochs))
        running_loss = 0.0

        for n_batch, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            if not lstm:
                tasks = torch.stack(batch['question']
                                    ).t().float().to(device)
            else:
                # Ideally this would be computed only once
                # rather than every epoch.
                tasks = torch.stack([text_util.string_to_vec(s)
                                     for s in batch['task']]
                                    ).to(device)
            target = batch['target'].to(device)
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
                        if not lstm:
                            tasks = torch.stack(batch['question']
                                                ).t().float().to(device)
                        else:
                            # Ideally this would be computed only once
                            # rather than every epoch.
                            tasks = torch.stack([text_util.string_to_vec(s)
                                                for s in batch['task']]
                                                ).to(device)

                        target = batch['target'].to(device)
                        types = batch['type']

                        loss, preds = model_.test(images, tasks, target)
                        c = (preds == target).squeeze()
                        test_loss += loss.item()

                        rel_correct += c[types == 0
                                         ].sum().cpu().data.tolist()
                        non_rel_correct += c[types == 1
                                             ].sum().cpu().data.tolist()
                        rel_total += c[types == 0].size()[0]
                        non_rel_total += c[types == 1].size()[0]

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

                # Save plot after every step to allow early aborting
                # if more progress is made, this is overwritten
                plt.plot(train_loss_history, label='training loss')
                plt.plot(test_loss_history, label='test loss')
                plt.plot(rel_acc_history, label='rel acc')
                plt.plot(non_rel_acc_history, label='non rel acc')
                plt.legend()
                plt.savefig('outputs/%s.png' % name)
                plt.close()
                np.save('outputs/%s.npy' % name, np.array([train_loss_history,
                                                           test_loss_history,
                                                           rel_acc_history,
                                                           non_rel_acc_history])
                        )
        torch.save(model_.state_dict(), 'outputs/%s.pk' % name)

    return


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
