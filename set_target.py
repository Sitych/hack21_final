import h5py
import pandas as pd
import os

data = pd.read_csv("target/target_1.csv",sep=';')
for fs in os.walk('bigsets'):
    for h5py_name in fs[2]:
        h5py_path = os.path.join('bigsets', h5py_name)
        with h5py.File(h5py_path, 'a') as f:
            try:
                f.create_dataset('target1', data=data[h5py_name.split('.')[0]])
            except KeyError:
                print("Error: ", h5py_name)