'''
Code that implements a simple heplied network for pattern recognition
'''
import numpy as np
from skimage.transform import resize
from skimage.filters import threshold_mean
from skimage.color import rgb2gray, rgba2rgb


class HopfieldNetwork:
    '''
    Class that implements a hopfield network for storing patterns.
    The data that can be passed to the network are images that
    represent the patterns of interest, they will be converted
    to grayscale, resized with shape (128, 128) and then flattened.
    Resizing is done to lighten the computational cost.
    To return to the square shape of the image you can use the reshape method.    
    '''

    def __init__(self, train_data, asyn=False):
        '''
        Classe constructor

        Parameters
        ----------
        train_data : list
            List of training dataset, each element of
            the list must be a 1darray
        asyn : bool, optional, deafult False
            Variable to switch between syncronus or asyncronus 
            update for the neurons of the network
        '''

        self.train_data = [self.preprocess(data) for data in train_data]
        self.n_data     = len(train_data)
        self.n_neurons  = self.train_data[0].shape[0]
        self.W          = np.zeros((self.n_neurons, self.n_neurons))
        self.asyn       = asyn

    def train(self):
        '''
        Function for the training of the network
        '''
        
        rho = np.sum([np.sum(t) for t in self.train_data]) / (self.n_data*self.n_neurons)
        
        # Hebbian rule
        for i in range(self.n_data):
            t = self.train_data[i] - rho
            self.W += np.outer(t, t)
        
        # Make diagonal element of self.W = 0
        self.W -= np.diag(np.diag(self.W))
        self.W /= self.n_data

    
    def predict(self, data, iteration=100, threshold=0, chunk=None):
        '''
        Function to compute the output of the network

        Parameters
        ----------
        data : list
            A dataset with the same format of train_data
        iteration : int, optional, default 20
            Number of iteration in time
        threshold : float
            A threshold for neuron activation, like bias
        chunk : Nonetype or int, optional
            If the network is created with asyn = True
            for each time step we update chunck neurons.
            The dealut value is n_neurons // 50

        Returns
        -------
        pred : list
            prediction of the network
        '''
        
        self.iter  = iteration
        self.thr   = threshold
        self.chunk = self.n_neurons//50 if chunk is None else chunk

        # Copy to avoid modifications
        p_data = [self.preprocess(d) for d in data]
        c_data = np.copy(p_data)
        
        # Define predict list
        pred = []
        for i in range(len(data)):
            pred.append(self.update(c_data[i]))
        
        return pred
    
    def update(self, init_s):
        '''
        Function for updateing the state of the network

        Parameters
        ----------
        init_s : 1darray
            current state
        
        Returns
        -------
        s : 1darray
            state of the network after self.iter iterations
        '''
        if self.asyn==False:
   
            # Compute initial state energy
            s = init_s
            e = self.energy(s)
            
            # Iteration
            for i in range(self.iter):
                # Update state
                s = np.sign(self.W @ s - self.thr)
                # Compute new state energy
                e_new = self.energy(s)

                # Check if the state has converged
                if e == e_new:
                    return s
                # Update energy
                e = e_new
            return s
        
        else:
   
            # Compute initial state energy
            s = init_s
            e = self.energy(s)
            
            # Iteration
            for i in range(self.iter):
                for j in range(self.chunk):
                    # Select random neuron
                    idx = np.random.randint(0, self.n_neurons) 
                    # Update state
                    s[idx] = np.sign(self.W[idx].T @ s - self.thr)
                
                # Compute new state energy
                e_new = self.energy(s)

                # Check if the state has converged
                if e == e_new:
                    return s
                # Update energy
                e = e_new
            return s


    def energy(self, s):
        '''
        Function that compute the energy of the state

        Parameters
        ----------
        s : 1darray
            state of the network
        
        Returns
        -------
         -1/2 \sum_{ij} s_i W_{ij} s_j + \sum_i s_i thr
        '''
        return -0.5 * s @ self.W @ s + np.sum(s * self.thr)


    def preprocess(self, img, width=128, hight=128):
        '''
        Preprocessing of data.
        Takes an image, converts it to grayscale, shifts the
        values to get -1, e, 1 and then flattens it into a 1D array

        Parameters
        ----------
        img : 2darray
            input data
        width : int, optional, defult 128
            Width for resizing img
        hight : int, optional, defult 128
            Hight for resizing img
        
        Returns
        -------
        prepro : 1darray
            fratten preprocessed img
        '''
        if img.shape[-1] == 4:
            img = rgba2rgb(img)

        if len(img.shape) == 3:
            img = rgb2gray(img)
        
        img = resize(img, (hight, width), mode='reflect')
        
        thresh = threshold_mean(img)
        
        # From original to -1, 1
        binary = img > thresh
        prepro = 2 * binary.astype(int) - 1

        # 1d vector
        return prepro.flatten()
    
    def reshape(self, img):
        '''
        Return to the original shape for visualizzation
        of the output of the nework

        Parameters
        ----------
        img : 1darray of size N^2
            output of the tework

        Return
        ------
        img : 2darray of shape (N, N)
            picture in grasy scale
        '''
        D   = int(np.sqrt(len(img)))
        img = np.reshape(img, (D, D))
        return img


