import numpy as np
import torch
#RLE编码
#把图像编码为RLE
#本题RLE 就是索引+个数，记录1的索引和1 后面有多少个1

im = torch.randn([2,2,3])
im = im.numpy()
im = im.flatten(order='F')
#展开
print(im.shape)
im = np.array([0,0,1,1,1,0,0,0,1,1,1,0,0,1,1])
im = im.reshape(3,5)
print(im)
im = im.flatten(order='C')
print(im)
im = np.concatenate([[0],im,[0]])
#首位添加0 [0，0,0,1,1,1,0,0,0,1,1,1,0,0,1,1，0] 和toch.cat类似
print(im.shape)
run = np.where(im[1:]!=im[:-1])[0]+1
#不加[0]返回一个list(array([ 2,  5,  8, 11, 13, 15]),)
print(run)
#得到1的索引
#个数5-2 11-8 15-3

print(run[1::2] -run[::2])
run[1::2]=run[1::2]-run[::2]
print(run)
str =" ".join(str(x) for x in run)
print(str)
#字符串用空格隔开
print(str.split())
s= str.split()
#---
starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]

print(starts,lengths)
starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
print(starts,lengths)
#---
#等价的
starts -=1
ends = starts + lengths
print(ends)
h=3
w=5

im = np.zeros((h*w),dtype=np.uint8)

for i,j in zip(starts,ends):
    im[i:j]=1
print(im)
im=im.reshape(h,w)
print(im)
