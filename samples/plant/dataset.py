import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
import split_folders

def label2masks(label_path, folder):

    label=cv2.imread(label_path)
    labelname=os.path.basename(label_path)
    id=labelname[:-10]
    label_gray=cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    leaf_num=list(np.unique(label_gray))
    if leaf_num[0]==0:
        leaf_num=leaf_num[1:]

    masks=[]
    for i in leaf_num:
        mask=np.where(label_gray==i,255,0).astype(np.uint8)
        maskid=str(i).zfill(3)
        maskname=id+'_'+maskid+'.png'
        maskpath=os.path.join(folder,maskname)
        cv2.imwrite(maskpath,mask)
        masks.append(mask)



def prepare_data(directory, dest):

    for filename in os.listdir(directory):
        if filename.endswith("_rgb.png"):
           name=filename[:-4]
           img_id=filename[:-8]
           label = img_id + '_label.png'
           label_path=os.path.join(directory,label)

           if os.path.isfile(os.path.join(directory,label)):
               imageFolder = os.path.join(dest, name)
               if not os.path.exists(imageFolder):
                  os.makedirs(imageFolder)
                  subimgFolder=os.path.join(imageFolder,'images')
                  os.makedirs(subimgFolder)
                  submaskFolder=os.path.join(imageFolder,'masks')
                  os.makedirs(submaskFolder)
                  shutil.copyfile(os.path.join(directory,filename), os.path.join(subimgFolder, filename))

                  label2masks(label_path, submaskFolder)



if __name__=='__main__':

    #readin_label()
    directory='/home/mary/AI/data/arabidopsis/synthetic_arabidopsis_dataset/synthetic_arabidopsis'
    dest='/home/mary/AI/data/arabidopsis/synthetic_arabidopsis_dataset/plant_data'
    split = '/home/mary/AI/data/arabidopsis/synthetic_arabidopsis_dataset/split'
    #prepare_data(directory, dest)

    files = os.listdir(dest)
    for f in files:
        if np.random.rand(1) < 0.2:
            shutil.move(dest + '/' + f, split + '/' + f)
