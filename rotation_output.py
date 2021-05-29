import numpy as np
from helper import eulerAnglesToRotationMatrix
import glob


for file in glob.glob('output/*.txt'):
    data = np.loadtxt(file, delimiter=',')
    full_list = []
    for row in data:
        R = eulerAnglesToRotationMatrix(row[:3])
        t = row[3:].reshape(3,1)
        R = np.append(R,t,axis=1)
        R = R.reshape(12,)
        full_list.append(R)
    np.savetxt(file,full_list,fmt='%.8f')