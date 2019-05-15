from CNN.forward import *
import numpy as np
from glob import glob
from PIL import Image

#####################################################
################## Utility Methods ##################
#####################################################
        
np.random.seed(42)
def extract_data(folder, IMAGE_WIDTH, m=200):
    '''
    Extract images by reading the file bytestream. Reshape the read values into a 3D matrix of dimensions [m, h, w], where m 
    is the number of training examples.
    '''
    print('Extracting', folder)
    files = list(glob(f'{folder}/**/*.jpg'))[:m]
    data = np.zeros((len(files), 3*IMAGE_WIDTH*IMAGE_WIDTH))
    for idx, x in enumerate(files):
        img = Image.open(x)
        img = img.resize((IMAGE_WIDTH,IMAGE_WIDTH), Image.ANTIALIAS)
        img = np.array(img)
        img = np.moveaxis(img, -1, 0)
        data[idx, :] = img.reshape(-1)

    return data

    
def extract_labels(folder, m=200):
    '''
    Extract label into vector of integer values of dimensions [m, 1], where m is the number of images.
    '''
    print('Extracting', folder)
    files = list(glob(f'{folder}/**/*.jpg'))[:m]
    labels = np.zeros((len(files), 1))
    for idx, x in enumerate(files):
            labels[idx, :] = int(x.split('/')[-2]) - 1
    
    return labels

def initializeFilter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale = stddev, size = size)

def initializeWeight(size):
    return np.random.standard_normal(size=size) * 0.01

def nanargmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs    

def predict(image, f1, f2, w3, w4, b1, b2, b3, b4, conv_s = 2, pool_f = 2, pool_s = 2):
    '''
    Make predictions with trained filters/weights. 
    '''
    conv1 = convolution(image, f1, b1, conv_s) # convolution operation
    conv1[conv1<=0] = 0 #relu activation
    
    conv2 = convolution(conv1, f2, b2, conv_s) # second convolution operation
    conv2[conv2<=0] = 0 # pass through ReLU non-linearity
    
    pooled = maxpool(conv2, pool_f, pool_s) # maxpooling operation
    (nf2, dim2, _) = pooled.shape
    fc = pooled.reshape((nf2 * dim2 * dim2, 1)) # flatten pooled layer
    
    z = w3.dot(fc) + b3 # first dense layer
    z[z<=0] = 0 # pass through ReLU non-linearity
    
    out = w4.dot(z) + b4 # second dense layer
    probs = softmax(out) # predict class probabilities with the softmax activation function
    
    return np.argmax(probs), np.max(probs)
    