import os
import numpy as np


def change_npy():
    path = f'/home/zyj/EventCSLR_gqr/to_delete'
    train_path = f'{path}/train_info.npy'
    dev_path = f'{path}/dev_info.npy'
    test_path = f'{path}/test_info.npy'
    train_data = np.load(train_path, allow_pickle=True).item()
    dev_data = np.load(dev_path, allow_pickle=True).item()
    test_data = np.load(test_path, allow_pickle=True).item()

    for key in train_data:
        if key == 'prefix':
            continue
        cur_dict = train_data[key]
        print(f'pre: {cur_dict}')
        cur_dict['folder'] = cur_dict['folder'].replace('/home/zyj/wangyh/dataset/EVSLRDataset','./data/EvCSLR')
        cur_dict['original_info'] = cur_dict['original_info'].replace('/home/zyj/wangyh/dataset/EVSLRDataset','./data/EvCSLR')
        print(f'aft: {cur_dict}')
    for key in dev_data:
        if key == 'prefix':
            continue
        cur_dict = dev_data[key]
        print(f'pre: {cur_dict}')
        cur_dict['folder'] = cur_dict['folder'].replace('/home/zyj/wangyh/dataset/EVSLRDataset','./data/EvCSLR')
        cur_dict['original_info'] = cur_dict['original_info'].replace('/home/zyj/wangyh/dataset/EVSLRDataset','./data/EvCSLR')
        print(f'aft: {cur_dict}')
    for key in test_data:
        if key == 'prefix':
            continue
        cur_dict = test_data[key]
        print(f'pre: {cur_dict}')
        cur_dict['folder'] = cur_dict['folder'].replace('/home/zyj/wangyh/dataset/EVSLRDataset','./data/EvCSLR')
        cur_dict['original_info'] = cur_dict['original_info'].replace('/home/zyj/wangyh/dataset/EVSLRDataset','./data/EvCSLR')
        print(f'aft: {cur_dict}')
    
    np.save(f'/home/zyj/EventCSLR_gqr/preprocess/EvCSLRDataset/train_info.npy', train_data)
    np.save(f'/home/zyj/EventCSLR_gqr/preprocess/EvCSLRDataset/dev_info.npy', dev_data)  
    np.save(f'/home/zyj/EventCSLR_gqr/preprocess/EvCSLRDataset/test_info.npy', test_data)


if __name__ == '__main__':
    change_npy()
