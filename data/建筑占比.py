import pandas as pd
import matplotlib.pyplot as plt
import cv2
from data.rle import rle_decode,rle_encode
newcsv = "*********/new_train_mask.csv"
path = "**********/tcdata/train/train/"
data = pd.read_csv(newcsv, header=None, names=['name', 'mask'])
image = data["name"].values[1:]
label = data["mask"].values[1:]

im_C0=[]
im_C1=[]
im_0_1= []
jl = 0
h=w=512
print(w)
print("all",len(label))
for k in range(len(label)):
    jl+=1
    print(jl)
    s= 0.0
    imm = label[k].split()

    for i in imm[1::2]:
        s+=float(i)
    s = s/(h*w)
    if s==0.0:
        im_C0.append(k)
    elif s==1.0:
        im_C1.append(k)
    else:
        im_0_1.append(s)

print("all",len(label))
print("C0",len(im_C0))
print("C1",len(im_C1))
ss=0.0
for i in im_0_1:
    ss+=i
ss=ss/len(im_0_1)
print("C_0_1",len(im_0_1),ss)
print(im_0_1[0])

def get_im_id(id):

    print(image[id])
    im = cv2.imread(path+image[id])
    lab = rle_decode(label[id])
    print(label[id])
    plt.subplot(1,2,1)
    plt.imshow(im)

    plt.subplot(1,2,2)
    plt.imshow(lab)
    plt.show()

for i in im_C1:
    get_im_id(i)

out:
原始数据：30000
无建筑数据：5204
清洗后： 24796
C0 0 
C1 0  （全为建筑的数据）
C_0_1 24796 0.19004847807621914   建筑在图片中占比
建筑像素在整个图片中平均：49807=0.19*512*512

