#
#                  ___====-_  _-====___
#            _--^^^#####//      \\#####^^^--_
#         _-^##########// (    ) \\##########^-_
#        -############//  |\^^/|  \\############-
#      _/############//   (@::@)   \\############\_
#     /#############((     \\//     ))#############\
#    -###############\\    (oo)    //###############-
#   -#################\\  / VV \  //#################-
#  -###################\\/      \//###################-
# _#/|##########/\######(   /\   )######/\##########|\#_
# |/ |#/\#/\#/\/  \#/\##\  |  |  /##/\#/  \/\#/\#/\#| \|
# `  |/  V  V  `   V  \#\| |  | |/#/  V   '  V  V  \|  '
#    `   `  `      `   / | |  | | \   '      '  '   '
#                     (  | |  | |  )
#                    __\ | |  | | /__
#                   (vvv(VVV)(VVV)vvv)
#
#            God bless me,         no bug!
#                         `=---='
#
#     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import time
import h5py
from PIL import Image
import hnswlib
import numpy as np
import os
import argparse
from extract_cnn_vgg16_keras import VGGNet

os.environ["CUDA_VISIBLE_DEVICES"]="2"
# read in indexed images' feature vectors and corresponding image names加载数据

ap = argparse.ArgumentParser()
ap.add_argument("-query", required=True,
                help="Path to query which contains image to be queried")

args = vars(ap.parse_args())


h5f = h5py.File('feature0416.h5', 'r')
imgNames = h5f['dataset_2'][:]
h5f.close()

num_elements = len(imgNames)
print('\n检索图片库的总数：{}张\n'.format(num_elements))
labels_index = np.arange(num_elements)
#EMBEDDING_SIZE = feats.shape[1]
p = hnswlib.Index(space = 'cosine', dim = 512) # possible options are l2, cosine or ip
print('&&&&&&&&&&&&&&&&&&&')
time0 = time.time()
p.load_index('index_one_million.idx')
time01 = time.time()
print('\n加载模型所用时间：{}秒\n'.format(time01-time0))
print("*******************")
p.set_ef(300) # ef should always be > k

model = VGGNet()
time_vgg2 = time.time()
print('\n启动VGG_NET用时：{}秒\n'.format(time_vgg2-time01))

# from extract_cnn_vgg16_keras import VGGNet
queryDir = args["query"]


# queryImg = mpimg.imread(queryDir)  # 读取图片
# plt.title("Query Image")
# plt.imshow(queryImg)
# plt.show()
# init VGGNet16 model
time2 = time.time()

# extract query image's feature, compute simlarity score and sort
queryVec = model.extract_feat(queryDir)
# time02 = time.time()
# print('\n启动配置环境 + 提取特征所用总时间：{}秒\n'.format(time02-time2))
time3 = time.time()
labels, distances = p.knn_query(queryVec, k = 100)
# print(distances)

imlist = [imgNames[index] for i, index in enumerate(labels)][0]
time03 = time.time()
print('\n检索100张相似图所用时间：{}秒\n'.format(time03-time3))
print('\n总用时间：{}秒\n'.format(time03-time2))
for i, im in enumerate(imlist):
        im=im.decode('utf-8')
        # image = mpimg.imread(im)
        img = Image.open(im)
        img.save('./1/%s.jpg' % i)
        # plt.title("search output %d" % (i + 1))
        # plt.imshow(image)
        # plt.show()
print('\n*****完成*****\n')
# # images = [queryImg]
# # images += [plt.imread(imgNames for label in labels[0])]
# plot_predictions(images)
