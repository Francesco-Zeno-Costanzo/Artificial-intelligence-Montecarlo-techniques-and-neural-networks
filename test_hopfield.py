'''
Code for testing hopfield.py code
'''
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from hopfield import HopfieldNetwork

np.random.seed(69420)

def add_noise(image, **kwargs):
    '''
    Function that add gaussina noise
    
    Parameters
    ----------
    image : Ndarray
        image as numpy array
    
    Return
    ------
    noisy_image : Ndarray
        noisy image as numpy array
    
    Other Parameters
    ----------------
    mean : float, optional, default 0
        mean of the gaussian
    std : float, optional, default 20
        standard deviation of the gaussian
    ampl : float, optional, default 20
        Amplitude of the gaussian
    '''
    mean = kwargs.get('mean',  0)
    std  = kwargs.get('std',  20)
    ampl = kwargs.get('ampl', 20)

    noise = np.random.normal(mean, std, image.shape)
    
    noisy_image = image + noise * ampl
    
    return noisy_image


def plot(data, test, pred, figsize=(8, 8)):
    '''
    function for plot

    Parameters
    ----------
    data : list
        list of orginal pattern
    test : list
        list of the input data for the nework
    pred : list
        prediction of the network
    '''
    
    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(len(data)):
        
        axarr[0, 0].set_title('Original data')
        axarr[0, 1].set_title("Input data")
        axarr[0, 2].set_title('Output data')
        
        axarr[1, 0].set_ylabel('train data')
        axarr[3, 0].set_ylabel('test data')


        axarr[i, 0].imshow(data[i])
        axarr[i, 1].imshow(test[i])
        axarr[i, 2].imshow(pred[i])

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    #===================== Dataset =====================#
    data = [Image.open('img1.jpg'),    # Three 
            Image.open("sfondo.png"),  # different
            Image.open("sfondo2.jpg"), # pictures
            Image.open('dsom.png')     # This is very similar to one of the previous ones
            ]
    # Convert in numpy array    
    data = [np.array(img) for img in data]
    
    #plot(data, test, data);exit()

    # We will use the first three pictures as train dataset
    train_data = data[:3]
    
    #===================== Train network =====================#
    
    net = HopfieldNetwork(train_data)#, asyn=True)
    net.train()

    #===================== Prediction =====================#

    test = [add_noise(d) for d in data]
    pred = net.predict(test)#, iteration=100, chunk=500)
    pred = [net.reshape(p) for p in pred]

    #===================== Plot =====================#

    # For simplicity. This is how the network see the data
    data = [net.reshape(net.preprocess(d)) for d in data]
    test = [net.reshape(net.preprocess(t)) for t in test]

    plot(data, test, pred)