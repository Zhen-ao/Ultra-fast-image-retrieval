import hnswlib
import numpy as np
import time
import hnswlib
import numpy as np
import h5py

# read in indexed images' feature vectors and corresponding image names加载数据
time0 = time.time()
h5f = h5py.File('feature0416.h5', 'r')

feats = h5f['dataset_1'][:]
imgNames = h5f['dataset_2'][:]
h5f.close()
print(feats.shape)


time01 = time.time()
print(time01-time0)

num_elements = len(feats)
print(num_elements)
labels_index = np.arange(num_elements)
EMBEDDING_SIZE = feats.shape[1]
# EMBEDDING_SIZE = 512
# num_elements = 50826900
time.sleep(5)
# Declaring index
p = hnswlib.Index(space = 'cosine', dim = EMBEDDING_SIZE) # possible options are l2, cosine or ip
print("****************")
# Initing index - the maximum number of elements should be known beforehand
p.init_index(max_elements = num_elements, ef_construction = 100, M = 16)
print("^^^^^^^^^^^^^^^^^^")
# Element insertion (can be called several times):
int_labels = p.add_items(feats, labels_index)
print("%%%%%%%%%%%%%%%%%%%%%")
# Controlling the recall by setting ef:
p.set_ef(100) # ef should always be > k
print("####################")
p.save_index('index_one_million.idx')
