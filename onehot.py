import numpy as np

def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    # print(buf.shape)
    nmsk = np.arange(data.size)*n + data.ravel()
    # print(nmsk)
    buf.ravel()[nmsk-1] = 1
    # print(buf.ravel().shape)
    return buf

if __name__ =='__main__':
    a=np.zeros((4,4)).astype('uint8')
    a=onehot(a,2)