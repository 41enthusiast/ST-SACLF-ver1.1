from glob import glob
import shutil 
import os

lblspath = '../datasets/dtd/labels/'
imgspath = '../datasets/dtd/images/'
tgtpath = '../datasets/dtd/imagenetstyle/'

for major_label in ['train', 'test', 'val']:
    for cls in os.listdir(imgspath):
        os.makedirs(tgtpath+major_label+'/'+cls, exist_ok=True) 
    for fname in glob(lblspath+major_label+'*'):
        with open(fname, 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                shutil.copyfile(imgspath+line, tgtpath+major_label+'/'+line)
