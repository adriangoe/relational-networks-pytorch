# -*- coding: utf-8 -*-
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

        # (sizes as in Appendix D of original paper)
        size0 = 32
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, size0, 3, stride=2, padding=1),
            nn.BatchNorm2d(size0),
            nn.ReLU()
        )
        size1 = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(size0, size1, 3, stride=2, padding=1),
            nn.BatchNorm2d(size1),
            nn.ReLU()
        )
        size2 = 128
        self.conv2 = nn.Sequential(
            nn.Conv2d(size1, size2, 3, stride=2, padding=1),
            nn.BatchNorm2d(size2),
            nn.ReLU()
        )
        self.size_out = 256
        self.conv3 = nn.Sequential(
            nn.Conv2d(size2, self.size_out, 3, stride=2, padding=1),
            nn.BatchNorm2d(self.size_out),
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


class BaseModel(nn.Module):
    '''Base class for both the RN
    and the MLP we use for comparisson.
    Using a base-class ensures consistency of
    learning-parameters and reduces complexity.
    '''

    def __init__(self, lstm=False, vocab=0):
        super(BaseModel, self).__init__()

        self.cnn = ConvolutionalInput()
        self.question_size = 10
        self.n_classes = 11
        self.lstm = lstm

        if self.lstm:
            # This is used to analyze the questions at input time.
            self.question_size = 20
            size_embedding = 10
            self.embedding = nn.Embedding(num_embeddings=vocab+1,
                                          embedding_dim=size_embedding)
            self.lstm_layer = nn.LSTM(input_size=size_embedding,
                                      hidden_size=self.question_size,
                                      batch_first=True)

        self.optimizer = None

    def train(self, images, questions, targets):
        '''Takes a batch of training data
        and performs backprop
        '''
        if not self.optimizer:
            self.setup_optimizer()
        self.optimizer.zero_grad()

        output = self(images, questions)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

        return loss, output.data.max(1)[1]

    def test(self, images, questions, targets):
        '''Takes question + image pairs
        and predicts the answer for each without
        training the model.
        '''
        output = self(images, questions)
        loss = F.cross_entropy(output, targets)

        return loss, output.data.max(1)[1]

    def setup_optimizer(self):
        '''Optimizer should be created when all parameters
        are ready and model is moved to the CUDA device
        '''
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)


class RelationalNetwork(BaseModel):
    def __init__(self, **args):
        super(RelationalNetwork, self).__init__(**args)
        # FUNCTION G:
        # input: object pair + question vector
        # output: flexible
        # (n_filters per object + coordinate_object) * 2 + question_vector
        # (sizes as in Appendix D of original paper)
        self.g_size = 2000
        self.g_theta = nn.Sequential(
            nn.Linear((self.cnn.size_out + 2) * 2 + self.question_size,
                      self.g_size),
            nn.BatchNorm1d(self.g_size),
            nn.ReLU(),
            nn.Linear(self.g_size, self.g_size),
            # nn.BatchNorm1d(self.g_size),
            nn.ReLU(),
            nn.Linear(self.g_size, self.g_size),
            # nn.BatchNorm1d(self.g_size),
            nn.ReLU(),
            nn.Linear(self.g_size, self.g_size),
            # nn.BatchNorm1d(self.g_size),
            nn.ReLU(),
        )

        # FUNCTION F:
        # takes element-wise sum of g-outputs
        # (sizes as in Appendix D of original paper)
        self.f_phi = nn.Sequential(
            nn.Linear(self.g_size, 1000),
            # nn.Dropout(),
            nn.ReLU(),
            nn.Linear(1000, 500),
            # nn.Dropout(),
            nn.ReLU(),
            nn.Linear(500, 100),
            # nn.Dropout(),
            nn.ReLU(),
            nn.Linear(100, self.n_classes),
            # nn.Dropout(),
            nn.LogSoftmax(dim=1),
        )

        #
        self.create_coordinates(25, 8)

    def forward(self, images, questions):
        x = self.cnn(images)

        if self.lstm:
            # Get hidden state as the question
            # embedding
            questions = self.embedding(questions)
            _, hidden = self.lstm_layer(questions)
            questions = hidden[0][0]

        '''Function g
        applied to pairs of object represenations
        '''

        # batch_size, representation_size, number_of_cells
        batch_size, k, d, _ = x.size()

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
        q_len = questions.size()[1]
        questions = torch.unsqueeze(questions, 1)
        questions = questions.repeat(1, d*d, 1)
        questions = torch.unsqueeze(questions, 2)

        # Combine all pairs of cells
        x_i = torch.unsqueeze(x, 1)
        # We repeat each cell, n_cells times along axis 1
        x_i = x_i.repeat(1, d*d, 1, 1)
        x_j = torch.unsqueeze(x, 2)
        # Inject the question with the second object
        x_j = torch.cat([x_j, questions], 3)
        # We repeat each cell, n_cells times along axis 2
        x_j = x_j.repeat(1, 1, d*d, 1)

        # Concatenate together along axis 3 to get all cell-pairs
        x = torch.cat([x_i, x_j], 3)

        # Expand: each cell gets its own row
        x = x.view(batch_size*d*d*d*d, (k+2)*2 + q_len)
        x = self.g_theta(x)

        # Reshape back to access individual images
        x = x.view(batch_size, d*d*d*d, self.g_size)
        x = x.sum(1).squeeze()

        '''Function f
        applied to combined relations to get output
        '''
        return self.f_phi(x)

    def create_coordinates(self, map_size, n_maps):
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
        coord_tensor = coord_tensor.unsqueeze(0).repeat(map_size, 1, 1, 1)

        coord_tensor = coord_tensor.view(
            map_size, 2, n_maps*n_maps).permute(0, 2, 1)

        self.coord_tensor = to_variable(coord_tensor)
        self.coord_tensor.requires_grad = False


class CNN_MLP(BaseModel):
    def __init__(self, **args):
        super(CNN_MLP, self).__init__(**args)

        self.cnn = ConvolutionalInput()

        # Inferred from description in Appendix D of
        # original paper.
        self.mlp = nn.Sequential(
            nn.Linear(self.cnn.size_out * 25 + self.question_size,
                      2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, self.n_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, images, questions):
        x = self.cnn(images)

        if self.lstm:
            # Get hidden state as the question
            # embedding
            questions = self.embedding(questions)
            _, hidden = self.lstm_layer(questions)
            questions = hidden[0][0]

        # Flatten output
        x = x.view(x.size(0), -1)
        # Add question to each image representation
        x = torch.cat((x, questions), 1)

        x = self.mlp(x)
        return x
