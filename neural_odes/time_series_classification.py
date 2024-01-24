######################
# So you want to train a Neural CDE model?
# Let's get started!
######################

#minimally modified from the example at https://github.com/patrick-kidger/torchcde.

import math
import torch
import torchcde
import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split

######################
# A CDE model looks like
#
# z_t = z_0 + \int_0^t f_\theta(z_s) dX_s
#
# Where X is your data and f_\theta is a neural network. So the first thing we need to do is define such an f_\theta.
# That's what this CDEFunc class does.
# Here we've built a small single-hidden-layer neural network, whose hidden layer is of width 128.
######################
class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        ######################
        # Actually solve the CDE.
        ######################
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.interval)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y

######################
# Next, we'll define a baseline model for comparison
######################
class RNN(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(RNN, self).__init__()

        self.hidden_dim=hidden_dim

        # define an RNN with specified parameters
        # batch_first means that the first dim of the input and output will be the batch_size
        self.rnn = torch.nn.RNN(input_size, hidden_dim, n_layers, batch_first=True, nonlinearity='relu')
        # last, fully-connected layer
        self.fc = torch.nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        # x (batch_size, seq_length, input_size)
        # hidden (n_layers, batch_size, hidden_dim)
        # r_out (batch_size, time_step, hidden_size)
        batch_size = x.size(0)

        # get RNN outputs
        r_out, hidden = self.rnn(x, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        r_out = r_out.view(-1, self.hidden_dim)

        # get final output
        output = self.fc(r_out)


        return output, hidden

######################
# Now we need some data.
# Here we have a simple example which generates some spirals, some going clockwise, some going anticlockwise.
######################
def get_data():
    ######################
    # X is a tensor of observations, of shape (batch=128, sequence=100, channels=3)
    # y is a tensor of labels, of shape (batch=128,), either 0 or 1 corresponding to anticlockwise or clockwise
    # respectively.
    ######################
    #print(train_X.shape) batch x timesteps x channels (where one is time)
    #print(train_y.shape) #the classifier
    #process the lightcurves:

    fns = glob.glob("/Users/alexgagliano/Documents/Conferences/FreedomTrail_Jan24/neural_odes/simulated_data/JieData_NoiseLess/*.ts")
    events = [int(x.split("/")[-1][:-3]) for x in fns]
    metadata = pd.read_csv("/Users/alexgagliano/Documents/Conferences/FreedomTrail_Jan24/neural_odes/simulated_data/noiseless_metadata.csv")

    y = metadata.iloc[np.argsort(events)]['numax'].values

    fulldata = []
    for fn in fns:
        fulldata.append(np.loadtxt(fn))
    fulldata = np.array(fulldata)
    #subsample by an oom
    fulldata = fulldata[:, ::2, :]

    #first! predicting numax from the observations
    X = fulldata
    #normalize data - time and flux
    nchannels = np.shape(X)[-1]
    X_norm = []
    for i in np.arange(nchannels):
        mean = (X[:, :, i]).mean()
        std = (X[:, :, i]).std()
        X_norm.append((X[:, :, i] - mean) / (std + 1e-5))
    X_norm = np.moveaxis(X_norm, 0, -1)

    #normalize so that each channel has mean 0 and variance 1
    #take 100 observations only
    X_train, X_test, y_train, y_test = train_test_split(X_norm[0:128], y[0:128], test_size=0.33, random_state=42)

    #try with a single batch for now...
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

def train_test_cde(model, train_X, train_y, test_X, test_y, num_epochs=30, lr=0.01, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ######################
    # Now we turn our dataset into a continuous path. We do this here via Hermite cubic spline interpolation.
    # The resulting `train_coeffs` is a tensor describing the path.
    # For most problems, it's probably easiest to save this tensor and treat it as the dataset.
    ######################
    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_X)

    train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)
            #loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, batch_y)
            #turn to an MSE loss for predicting numax
            loss = torch.nn.MSELoss()(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))

    test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_X)

    pred_y = model(test_coeffs).squeeze(-1)

    #plot the results for numax prediction
    #from https://github.com/patrick-kidger/torchcde/tree/master
    plt.figure(figsize=(10,7))
    plt.plot(test_y.detach.numpy(), pred_y.detach.numpy(), 'o')
    plt.xlabel("True numax")
    plt.ylabel("Predicted numax")
    plt.title("Neural (C)-Differential Equation")
    print("CDE loss = ", torch.nn.MSELoss()(test_y, pred_y))

def train_test_rnn(model, train_X, train_y, test_X, test_y, num_epochs=30, lr=0.01, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch_X, batch_y = batch
            pred_y = model(batch_X).squeeze(-1)
            #loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, batch_y)
            #turn to an MSE loss for predicting numax
            loss = torch.nn.MSELoss()(pred_y, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))

    pred_y = model(test_y).squeeze(-1)

    #plot the results for numax prediction
    plt.figure(figsize=(10,7))
    plt.plot(test_y.detach.numpy(), pred_y.detach.numpy(), 'o')
    plt.xlabel("True numax")
    plt.ylabel("Predicted numax")
    plt.title("Traditional RNN")
    print("Traditional RNN loss = ", torch.nn.MSELoss()(test_y, pred_y))


#def main(num_epochs=30):
train_X, train_y, test_X, test_y = get_data()
######################
# input_channels=2 here because we only have flux, and time.
# hidden_channels=8 is the number of hidden channels for the evolving z_t, which we get to choose.
# output_channels=1 because we're doing numax prediction.
######################
model_cde = NeuralCDE(input_channels=2, hidden_channels=8, output_channels=1)
model_rnn = RNN(input_size=2, output_size=1, hidden_dim=128, n_layers=1)

train_test_cde(model_cde, train_X, train_y, test_X, test_y)
train_test_rnn(model_rnn, train_X, train_y, test_X, test_y)



#if __name__ == '__main__':
#    main()
