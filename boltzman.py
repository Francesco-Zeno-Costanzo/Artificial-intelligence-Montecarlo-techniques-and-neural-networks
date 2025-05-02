import numpy as np

np.random.seed(69420)

class RBM:
   
    def __init__(self, visible_dim, hidden_dim):
        '''
        Restricted Boltzmann Machine (RBM) using Contrastive Divergence (CD-1).

        Parameters
        ----------
        visible_dim : int
            Number of visible units (input features).
        hidden_dim : int
            Number of hidden units (latent features).
        '''
        self.visible_dim = visible_dim
        self.hidden_dim  = hidden_dim

        # Initialize weights and biases
        self.W  = np.random.randn(visible_dim, hidden_dim) * 0.1 # Weights
        self.bh = np.zeros(hidden_dim)                           # Hidden bias
        self.bv = np.zeros(visible_dim)                          # Visible bias


    def gen_batches(self, n_samples, batch_size):
        '''
        Function to generate the mini-batch for training

        Parameters
        ----------
        n_samples : int
            number of input data
        batch_size : int
            size of the mini-batch
        '''
        for i in range(0, n_samples, batch_size):
            yield slice(i, min(i + batch_size, n_samples))
    

    def sigmoid(self, x):
        ''' Activation function
        '''
        return 1 / (1 + np.exp(-x))


    def fit(self, X, epochs=100, batch_dim=16, lr=0.01, verbose=False):
        '''
        Trains the RBM using Contrastive Divergence (CD-1).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data (should be binary or normalized in [0,1]).
        epochs : int, default=100
            Number of training iterations over the dataset.
        batch_dim : int, default=16
            Size of each mini-batch.
        lr : float, default=0.01
            Learning rate.
        Verbose : bool, optional, default False
            If True print error each epoch
        
        Description
        -----------
        This method implements the Contrastive Divergence (CD-1) algorithm to approximate
        the gradient of the log-likelihood of the data. The procedure is:

        1. Positive Phase:
            - Compute P(h|v0): hidden unit activation probabilities given input v0.
            - Compute the "positive" gradient: v0.T @ P(h|v0)

        2. Negative Phase:
            - Sample h0 from P(h|v0)
            - Reconstruct visible units: sample v1 from P(v|h0)
            - Compute P(h|v1)
            - Compute the "negative" gradient: v1.T @ P(h|v1)

        3. Update Parameters:
            - Weights:         W += lr * (positive_grad - negative_grad) / batch_size
            - Visible biases: bv += lr * mean(v0 - v1)
            - Hidden biases:  bh += lr * mean(P(h|v0) - P(h|v1))

        '''
        n_samples = X.shape[0]

        for epoch in range(epochs):
            
            error_epoch = 0
            
            for slice in self.gen_batches(n_samples, batch_dim):
                
                batch      = X[slice]
                batch_size = batch.shape[0]

                # -------- Positive phase --------
                # P(h=1 | v)
                # Compute the probability that each hidden neuron fires given the visible input.
                pos_hid_probs = self.sigmoid(np.dot(batch, self.W) + self.bh)
                # Correlation between visible and hidden
                pos_grad      = np.dot(batch.T, pos_hid_probs)

                # -------- Negative phase --------
                # Sample hidden states from positive hidden probabilities
                pos_hid_states = pos_hid_probs > np.random.rand(*pos_hid_probs.shape)
                # P(v' = 1 | h)
                reconstructed  = self.sigmoid(np.dot(pos_hid_states, self.W.T) + self.bv)
                # P(h' = 1 | v')
                neg_hid_probs  = self.sigmoid(np.dot(reconstructed, self.W) + self.bh)
                # Correlation for negative data
                neg_grad       = np.dot(reconstructed.T, neg_hid_probs)

                # -------- Parameter update --------
                self.W  += lr * (pos_grad - neg_grad) / batch_size
                self.bv += lr * np.mean(batch - reconstructed, axis=0)
                self.bh += lr * np.mean(pos_hid_probs - neg_hid_probs, axis=0)

                # -------- Error computation --------
                error = np.mean((batch - reconstructed) ** 2)
                error_epoch += error

            error_epoch /= batch_dim
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - error: {error_epoch:.7f}")


    def reconstruct(self, X):
        '''
        Reconstructs input data by encoding and decoding through the hidden layer.

        Parameters
        ----------
        X : 2darray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        reconstructed : 2darray, same shape of X
            Reconstructed visible data.
        '''
        # Encode
        hidden_probs  = self.sigmoid(np.dot(X, self.W) + self.bh)
        hidden_states = hidden_probs > np.random.rand(*hidden_probs.shape)

        # Decode
        reconstructed = self.sigmoid(np.dot(hidden_states, self.W.T) + self.bv)
        return reconstructed


if __name__ == "__main__":

    import numpy as np
    import matplotlib.pyplot as plt


    path = '/home/francesco/GitHub/neural-network/MNIST_data/'


    #================== Train Data ==================#

    #  All 60000 data is too much
    N = 3000
    train_data = np.loadtxt(f'{path}mnist_train.csv', max_rows=N, delimiter=',')
    # Normalize input 
    X_train, Y_train = train_data[:, 1:]/255, train_data[:, 0]
    Y_train = np.array([int(y) for y in Y_train])
    # Binary input
    X_train = np.where(X_train > 0.5, 1, 0)

    #================== Test Data ==================#

    M = 10000
    test_data = np.loadtxt(f'{path}/mnist_test.csv', max_rows=M, delimiter=',')
    # Normalize input 
    X_test, Y_test = test_data[:, 1:]/255, test_data[:, 0]
    Y_test = np.array([int(y) for y in Y_test])
    # Binary input
    X_test = np.where(X_test > 0.5, 1, 0)

    #================== Train ==================#

    rbm = RBM(visible_dim=X_train.shape[1], hidden_dim=100)
    rbm.fit(X_train, epochs=100, batch_dim=128, lr=0.1, verbose=True)

    #================== Test ==================#

    idx           = np.array([np.where(Y_test == i)[0][:3] for i in range(10)]).T.flatten()
    plot_data     = X_test[idx]
    reconstructed = rbm.reconstruct(plot_data)

    rows, cols = 6, 10

    fig, axes = plt.subplots(rows, cols, figsize=(8, 8))
    fig.suptitle("RBM Reconstruction on MNIST Digits", fontsize=18)

   
    for i in range(rows):
        for j in range(cols):
            idx_img = j + cols * (i // 2)
            ax = axes[i, j]
            if i % 2 == 0:
                img = plot_data[idx_img].reshape(28, 28)
            else:
                img = reconstructed[idx_img].reshape(28, 28)

            ax.imshow(img, cmap='gray')
            ax.tick_params(left=False, right=False, labelleft=False,
                    labelbottom=False, bottom=False)

    for i in [0, 2, 4]:
        axes[i,   0].set_ylabel("Actual",   fontsize=14)
        axes[i+1, 0].set_ylabel("Reconst.", fontsize=14)

    plt.tight_layout()
    plt.show()



   
    
