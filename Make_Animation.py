import numpy as np
from PIL import Image
import imageio
import os
output_name = 'disparity_graph.gif'

idx = 1
#for i in os.listdir('./asset/error_of_disparity_graph/'):
#    os.rename('line'+str(idx)+'disparity')


imglist = []
for i in os.listdir('./asset/error_of_disparity_graph/'):
    imglist.append(os.getcwd()+"/asset/error_of_disparity_graph/"+i)

print(imglist)

imglist.sort(key=lambda x : os.path.getmtime(x))
#images = list(map(lambda filename: imageio.imread(filename), imglist))

print(imglist)
images = list(imageio.imread(filename) for filename in imglist)

imageio.mimsave(os.path.join(output_name), images, duration=0.1)


