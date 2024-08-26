import os
import numpy as np

path = '../gloss_dict.txt'
file = open(path)
save_path = '../gloss_dict_new.txt'
save_file = open(save_path, 'a+')

dict_ = file.readline()

print(dict_)
for f in dict_.split('],'):
    save_file.write(f)
    save_file.write('\n')
