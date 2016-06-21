import numpy as np
import matplotlib.pyplot as plt
data_all = np.zeros((0,500))

for i in range(0,10):
    fname = 'proc'+str(i)+'snapshot9000.dat'
    dat = np.loadtxt(fname);
    print data_all.shape
    print dat.shape
    data_all = np.vstack((data_all,dat));
plt.imshow(X = data_all,vmin = -0.1, vmax = 0.1)
plt.colorbar()
plt.savefig('wavefield.png')
