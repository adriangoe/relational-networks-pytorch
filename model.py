import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


def to_variable(x):
    '''Utility function to make it
    easier to define GPU Variables
    based on the environment.
    '''
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class ConvolutionalInput(nn.Module):

    def __init__(self):
        '''The Convolutional part of the model
        takes in 3-channel images and turns them
        into a representation of contained objects.
        '''
        super(ConvolutionalInput, self).__init__()
        size = 24

        self.conv0 = nn.Sequential(
            nn.Conv2d(3, size, 3, stride=2, padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(size, size, 3, stride=2, padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(size, size, 3, stride=2, padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(size, size, 3, stride=2, padding=1),
            nn.BatchNorm2d(size),
            nn.ReLU()
        )

    def forward(self, img):
        '''Forward pass through
        the convolutional layers
        '''
        x = self.conv0(img)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


class FullyConnected(nn.Module):

    def __init__(self, n_out):
        super(FullyConnected, self).__init__()
        n_hidden = 256

        self.fc0 = nn.Sequential(
            nn.Linear(256, n_hidden),
            nn.ReLU(),
            nn.Dropout()
        )
        self.out = nn.Sequential(
            nn.Linear(n_hidden, n_out),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.fc0(x)
        return self.out(x)


class RelationalNetwork(nn.Module):
    def __init__(self):
        super(RelationalNetwork, self).__init__()

        self.cnn = ConvolutionalInput()

        cnn_size = 24
        question_size = 10
        n_classes = 11
        batch_size = 64
        learning_rate = 0.0001

        # FUNCTION G:
        # input: object pair + question vector
        # output: flexible
        # (n_filters per object + coordinate_object) * 2 + question_vector
        self.g_theta = nn.Sequential(
            nn.Linear((cnn_size + 2) * 2 + question_size,
                      256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
        )

        # FUNCTION F:
        # takes element-wise sum of g-outputs
        self.f_phi = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        self.out = FullyConnected(n_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.create_coordinates(batch_size, 8)

    def forward(self, img, qst):
        x = self.cnn(img)

        '''Function g
        applied to pairs of object represenations
        '''

        # batch_size, representation_size, number_of_cells
        batch_size, k, d, _ = x.size()
        # 64, 24, 8, 8

        # Stack all cells of represenations
        # to get (batch_size, cells, representation)
        x = x.view(batch_size, k, d * d).permute(0, 2, 1)

        # In case batch-size changes
        if self.coord_tensor.size()[0] != batch_size:
            self.create_coordinates(batch_size, d)

        # Add coordinates to each object
        x = torch.cat([x, self.coord_tensor], 2)
        # 64, 64, 26

        # Create question vector for each cell
        q_len = qst.size()[1]
        qst = torch.unsqueeze(qst, 1)
        qst = qst.repeat(1, d*d, 1)
        qst = torch.unsqueeze(qst, 2)

        # Combine all pairs of cells
        x_i = torch.unsqueeze(x, 1)
        # We repeat each cell, n_cells times along axis 1
        x_i = x_i.repeat(1, d*d, 1, 1)
        x_j = torch.unsqueeze(x, 2)
        # Inject the question with the second object
        x_j = torch.cat([x_j, qst], 3)
        # We repeat each cell, n_cells times along axis 2
        x_j = x_j.repeat(1, 1, d*d, 1)

        # Concatenate together along axis 3 to get all cell-pairs
        x = torch.cat([x_i, x_j], 3)

        # Expand: each cell gets its own row
        x = x.view(batch_size*d*d*d*d, (k+2)*2 + q_len)
        x = self.g_theta(x)

        # Reshape back to access individual images
        x = x.view(batch_size, d*d*d*d, 256)
        x = x.sum(1).squeeze()

        '''Function f
        applied to combined relations
        '''
        x = self.f_phi(x)

        # Pass through final dense layers to map
        # from inferred relation to output.
        return self.out(x)

    def create_coordinates(self, batch_size, n_maps):
        '''We create an arbitrary coordinate for each
        of the d**2 k-dimensional cells in the d*d feature
        maps indicating their relative spatial position.
        (see Section 4, dealing with pixels)
        '''
        coords = torch.linspace(-n_maps/2., n_maps/2., n_maps)
        x = coords.unsqueeze(0).repeat(n_maps, 1)
        y = coords.unsqueeze(1).repeat(1, n_maps)
        coord_tensor = torch.stack((x, y))

        # Create coordinates for each input
        coord_tensor = coord_tensor.unsqueeze(0
            ).repeat(batch_size, 1, 1, 1)

        coord_tensor = coord_tensor.view(
            batch_size, 2, n_maps*n_maps).permute(0, 2, 1)

        self.coord_tensor = to_variable(coord_tensor)
        self.coord_tensor.requires_grad = False


    def train(self, images, questions, targets):
        '''Takes a batch of training data
        and performs backprop
        '''
        self.optimizer.zero_grad()

        output = self(images, questions)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

        return loss, output.data.max(1)[1]

    def predict(self, images, questions):
        '''Takes question + image pairs
        and predicts the answer for each.
        '''
        output = self(images, questions)

        return output.data.max(1)[1]
