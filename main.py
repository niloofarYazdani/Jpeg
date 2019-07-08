# 4 steps of the JPEG encoding chain are demonstrated. These 4 steps contain the lossy parts of the coding. The remaining steps, i.e. runlength and Huffman encoding, are losless. Therefore the 4 steps demonstrated here are sufficient to study the quality-loss of JPEG encoding.
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm

#Step 1: Read & Display image
B=8     #block size
img1 = cv2.imread("harry.jpg")
print(img1.size)
print (img1.dtype)
h,w=np.array(img1.shape[:2])/B * B
img1=img1[:int(h),:int(w)]
#convert BGR to RGB     //In OpenCV color images are imported as BGR. In order to display the image with pyplot.imshow() the channels must be switched such that the order is RGB.
img2=np.zeros(img1.shape,np.uint8)  #shape & dataType
img2[:,:,0]=img1[:,:,2]
img2[:,:,1]=img1[:,:,1]
img2[:,:,2]=img1[:,:,0]
plt.imshow(img2)
cv2.imwrite("img2.jpg",np.uint8(img2))
print(img2.size)
#click into the image then DCT cofficients will be displayed later.
point=plt.ginput(1)
block=np.floor(np.array(point)/B)
print("Croodinates of selected block: ",block)
scol=block[0,0]
srow=block[0,1]
plt.plot([B*scol,B*scol+B,B*scol+B,B*scol,B*scol],[B*srow,B*srow,B*srow+B,B*srow+B,B*srow])
plt.axis([0,w,h,0])

#Step 2: Transform BGR to YCrCb and Subsample Chrominance Channels
#Subsample:meaning that color information is in lesser resolution than the luminance (greyscale) data.//show image with less pixel.//convert continuos signal to discrete signal
transcol=cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
#Subsampling
SSV=2       #Subsampling factor in vertical direction
SSH=2       #Subsampling factor in horicontal direction
crf=cv2.boxFilter(transcol[:,:,1],ddepth=-1,ksize=(2,2))    #Before subsampling the chrominance channels are filtered using a (2x2) box filter (=average filter)
cbf=cv2.boxFilter(transcol[:,:,2],ddepth=-1,ksize=(2,2))
crsub=crf[::SSV,::SSH]      #Subsampling
cbsub=cbf[::SSV,::SSH]
imSub=[transcol[:,:,0],crsub,cbsub]     #Stored all 3 channels


#Step 3 and 4: Discrete Cosinus Transform and Quantisation
#First the quantisation matrices for the luminace channel (QY) and the chrominance channels (QC) are defined, as proposed in the annex of the Jpeg standard.
QY=np.array([[16,11,10,16,24,40,51,61],
                         [12,12,14,19,26,48,60,55],
                         [14,13,16,24,40,57,69,56],
                         [14,17,22,29,51,87,80,62],
                         [18,22,37,56,68,109,103,77],
                         [24,35,55,64,81,104,113,92],
                         [49,64,78,87,103,121,120,101],
                         [72,92,95,98,112,100,103,99]])

QC=np.array([[17,18,24,47,99,99,99,99],
                         [18,21,26,66,99,99,99,99],
                         [24,26,56,99,99,99,99,99],
                         [47,66,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99],
                         [99,99,99,99,99,99,99,99]])

#
QF=30.0
if QF < 50 and QF > 1:
        scale = np.floor(5000/QF)
elif QF < 100:
        scale = 200-2*QF
else:
        print ("Quality Factor must be in the range [1..99]")
scale=scale/100.0
Q=[QY*scale,QC*scale,QC*scale]      #The list Q contains the 3 scaled quantisation matrices, which will be applied to the DCT coefficients:
#DCT and quantisation performe
TransAll=[]
TransAllQuant=[]
ch=['Y','Cr','Cb']
plt.figure()
for idx,channel in enumerate(imSub):
        plt.subplot(1,3,idx+1)
        channelrows=channel.shape[0]
        channelcols=channel.shape[1]
        Trans = np.zeros((channelrows,channelcols), np.float32)
        TransQuant = np.zeros((channelrows,channelcols), np.float32)
        blocksV=channelrows/B
        blocksH=channelcols/B
        vis0 = np.zeros((channelrows,channelcols), np.float32)
        vis0[:channelrows, :channelcols] = channel
        vis0=vis0-128
        for row in range(int(blocksV)):
                for col in range(int(blocksH)):
                        currentblock = cv2.dct(vis0[row*B:(row+1)*B,col*B:(col+1)*B])
                        Trans[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
                        TransQuant[row*B:(row+1)*B,col*B:(col+1)*B]=np.round(currentblock/Q[idx])
        TransAll.append(Trans)
        TransAllQuant.append(TransQuant)
        if idx==0:
                selectedTrans=Trans[int(srow*B):int((srow+1)*B),int(scol*B):int((scol+1)*B)]
        else:
                sr=np.floor(srow/SSV)
                sc=np.floor(scol/SSV)
                selectedTrans=Trans[int(sr*B):int((sr+1)*B),int(sc*B):int((sc+1)*B)]
        plt.imshow(selectedTrans,cmap=cm.jet,interpolation='nearest')
        plt.colorbar(shrink=0.5)
        plt.title('DCT of '+ch[idx])

#Decode

DecAll=np.zeros((int(h),int(w),3), np.uint8)
for idx,channel in enumerate(TransAllQuant):
        channelrows=channel.shape[0]
        channelcols=channel.shape[1]
        blocksV=channelrows/B
        blocksH=channelcols/B
        back0 = np.zeros((channelrows,channelcols), np.uint8)
        for row in range(int(blocksV)):
                for col in range(int(blocksH)):
                        dequantblock=channel[row*B:(row+1)*B,col*B:(col+1)*B]*Q[idx]
                        currentblock = np.round(cv2.idct(dequantblock))+128
                        currentblock[currentblock>255]=255
                        currentblock[currentblock<0]=0
                        back0[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
        back1=cv2.resize(back0,(int(h),int(w)))
        DecAll[:,:,idx]=np.round(back1)
#
reImg=cv2.cvtColor(DecAll, cv2.COLOR_YCrCb2BGR)
#cv2.cv.SaveImage('BackTransformedQuant.jpg', cv2.cv.fromarray(reImg))
print(reImg.SIZE)
cv2.imwrite("Finaly.jpg",np.uint8(reImg))
plt.figure()
img3=np.zeros(img1.shape,np.uint8)
img3[:,:,0]=reImg[:,:,2]
img3[:,:,1]=reImg[:,:,1]
img3[:,:,2]=reImg[:,:,0]
plt.imshow(img3)
SSE=np.sqrt(np.sum((img2-img3)**2))
print ("Sum of squared error: ",SSE)
plt.show()
