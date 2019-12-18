import imageio
from os import walk

f = []
for (dirpath, dirnames, filenames) in walk('figures/test'):
    f.extend(filenames)
    break
#print(f)
filenames = list(map(lambda x: int(x[:-4]), f))
filenames.sort()

images = []
for filename in filenames:
    images.append(imageio.imread('figures/test/%i.png' % (filename)))
imageio.mimsave('figures/test/test.gif', images, format='GIF', duration=1/60)
