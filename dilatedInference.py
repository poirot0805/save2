import time
import os
from PIL import Image,ImageFont,ImageDraw
from tqdm import tqdm
from collections import Counter
import pandas as pd
import cv2 as cv
import numpy as np
import pickle
import torch
from mmdet.apis import init_detector, inference_detector
from mmdet.core.visualization import imshow_det_bboxes
import mmcv

MYCLASSES = ('坐便器','小便器','蹲便器','台式洗脸盆','台式洗脸盆-双盆','台式洗脸盆-单盆',
        '长条台式洗脸盆','立式洗脸盆','洗脸盆','洗涤槽','洗涤槽-双槽','拖把池',
        '水龙头','洗衣机', '淋浴房', '淋浴房-转角型', '浴缸','淋浴器')

def get_clips(image,img_scale,step):
    """
    大图从左到右、从上到下滑窗切成块
    """
    clip_list=[]
    target_size=(img_scale,img_scale)   # 滑窗大小
    center_size=(step,step)
    cnt = 0
    target_w,target_h = target_size
    center_w,center_h = center_size
    h,w = image.shape[0],image.shape[1]
    # 填充至整数
    new_w = (w//center_w+1)*center_w
    new_h = (h//center_h+1)*center_h
    image = cv.copyMakeBorder(image,0,new_h-h,0,new_w-w,cv.BORDER_CONSTANT,value=(255,255,255))

    # 填充1/2 stride长度的外边框
    stride = img_scale-step
    h,w = image.shape[0],image.shape[1]
    new_w,new_h = w + stride,h + stride
    image = cv.copyMakeBorder(image,stride//2,stride//2,stride//2,stride//2,cv.BORDER_CONSTANT,value=(255,255,255))
    # crop
    h,w = image.shape[0],image.shape[1] # 新的长宽
    for j in range(h//step):
        for i in range(w//step):
            topleft_x = i*step
            topleft_y = j*step
            crop_image = image[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w]

            if crop_image.shape[:2]!=(target_h,target_h):
                print(topleft_x,topleft_y,crop_image.shape)

            else:
                clip_list.append(crop_image)
                # cv.imwrite("d:\\test\\"+str(cnt)+".jpg",crop_image)
                cnt+=1
    return clip_list

def create_zeros_png(image_w,image_h,img_scale,step):
    '''Description:
        0. 先创造一个空白图像，将滑窗预测结果逐步填充至空白图像中；
        1. 填充右下边界，使原图大小可以杯滑动窗口整除；
        2. 膨胀预测:预测时，对每个(400,400)窗口，每次只保留中心(300,300)区域预测结果，每次滑窗步长为300，使预测结果不交叠；
    '''
    margin=(img_scale-step)//2
    new_h,new_w = (image_h//step+1)*step,(image_w//step+1)*step #填充右边界
    zeros = (new_h,new_w,3)  
    print(zeros)
    zeros = np.zeros(zeros,np.uint8)
    return zeros

def get_inference_block_mask(bboxes,labels,scores,img_scale,id_start):
    """
    单个图块根据bboxes数据生成mask，每个像素只允许存在一种：生成的mask有categories——mask // score_mask // box_id_mask
    FIXME: box之间可能存在包含关系
    """
    mask_size=(img_scale,img_scale)
    temp_cat=np.zeros(mask_size,np.uint8)
    temp_id=np.zeros(mask_size,np.int32)
    temp_score=np.zeros(mask_size,np.float32)
    box_dict={}
    def myclamp(x):
        x=max(x,0)
        x=min(x,img_scale)
        return x
    for i, (bbox, label) in enumerate(zip(bboxes, labels)):
        id_start+=1
        score=1
        bbox_int = bbox.astype(np.int32)
        x1,y1,x2,y2=bbox_int[0], bbox_int[1],bbox_int[2], bbox_int[3]
        if len(bbox) > 4:
            score=bbox[-1]
        box_dict[id_start]=[x1,y1,x2,y2,label,score]

        x1=myclamp(x1)
        x2=myclamp(x2+1)
        y1=myclamp(y1)
        y2=myclamp(y2+1)
 
        for y in range(y1,y2):
            for x in range(x1,x2):
                if(temp_score[y,x]<score ):  # FIXME: 如果是同类别的话就不要更新了
                    temp_score[y,x]=score
                    temp_cat[y,x]=label+1   # FIXME:防止0
                    temp_id[y,x]=id_start     
    
    return temp_cat,temp_score,temp_id,id_start,box_dict

def merge_clip_top_and_left(thisclip_ids,mask_category,mask_id,step,thres):
    """
    衔接当前块与上方块&左方块
    """
    row_id,col_id=thisclip_ids
    this_x = col_id*step
    this_y = row_id*step
    temp=np.zeros((1,step),np.uint8)
    # top
    if row_id>0:
        top_x=this_x
        top_y=this_y-1
        inter_cnt=0
        for i in range(step):
            if(mask_category[this_y,this_x+i]==mask_category[top_y,top_x+i] and mask_category[this_y,this_x+i]!=0): # 边缘交集：相交=1
                temp[0,i]=1
                inter_cnt+=1
        i=0
        j=0
        pair_list=[]
        while(i<step):
            j=i
            while(temp[0,j]==1):
                j+=1
                if(j==step):
                    break
            if j!=i:
                pair_list.append((i,j-1))
            i=j
            i+=1
        
        for i in range(len(pair_list)):
            first,last=pair_list[i]
            length=last-first+1
            cat=mask_category[this_y,this_x+first]
            while(mask_category[top_y,top_x+first]==cat and first>=0):
                first-=1
            while(mask_category[top_y,top_x+last]==cat and last<step):
                last+=1
            first+=1
            last-=1
            if length/(last-first+1)>=thres:
               
                top_id=mask_id[top_y,top_x+(last+first)//2]
                first,last=pair_list[i]
                this_id=mask_id[this_y,this_x+(last+first)//2]
                flag=False
                for y in range(this_y,this_y+step):
                    for x in range(this_x,this_x+step):
                        if(mask_id[y,x]==this_id and mask_category[y,x]==cat):
                            flag=True
                            mask_id[y,x]=top_id

    # left
    temp=np.zeros((1,step),np.uint8)
    if col_id>0:
        top_x=this_x-1
        top_y=this_y
        inter_cnt=0
        for i in range(step):
            if(mask_category[this_y+i,this_x]==mask_category[top_y+i,top_x] and mask_category[this_y+i,this_x]!=0): # 相交
                temp[0,i]=1
        i=0
        j=0
        pair_list=[]    # 0-1串中连续1的起止
        while(i<step):
            j=i
            while(temp[0,j]==1):
                j+=1
                if j==step:
                    break
            if j!=i:
                pair_list.append((i,j-1))
            i=j
            i+=1
        for i in range(len(pair_list)):
            first,last=pair_list[i]
            length=last-first+1
            cat=mask_category[this_y+first,this_x]
            while(mask_category[top_y+first,top_x]==cat and first>=0):
                first-=1
            while(mask_category[top_y+last,top_x]==cat and last<step):
                last+=1
            first+=1
            last-=1
            if length/(last-first+1)>=thres:
                flag=False
                top_id=mask_id[top_y+(last+first)//2,top_x]
                # print("left yes!left_box_id:{}".format(top_id))
                first,last=pair_list[i]
                this_id=mask_id[this_y+(last+first)//2,this_x]
                for y in range(this_y,this_y+step):
                    for x in range(this_x,this_x+step):
                        if(mask_id[y,x]==this_id and mask_category[y,x]==cat):
                            flag=True
                            mask_id[y,x]=top_id
                
    return mask_category,mask_id
    
def clip_for_inference(model,clip_list,jpg_shape,img_scale,step,score_thres=0.5,device=None):
    """
    将clip_list中的clip逐个推理，并合并结果
    """
    image_h,image_w = jpg_shape
    predict_png = create_zeros_png(image_w,image_h,img_scale,step)
    h=predict_png.shape[0]  # 补整后
    w=predict_png.shape[1]
    mask_category=np.zeros((h,w),np.uint8)  # 类别
    mask_score=np.zeros((h,w),np.float32)   # 分数
    mask_id=np.zeros((h,w),np.int32)    # box_id
    dict_for_clips={}

    clip_cnt=0
    box_cnt=0
    margin=(img_scale-step)//2

    ALL_WHITE=img_scale*img_scale*3*255 # 全白不推理
    for j in tqdm(range(h//step)):
        for i in range(w//step):
            topleft_x = i*step
            topleft_y = j*step
            if np.sum(clip_list[clip_cnt])<ALL_WHITE:
                result = inference_detector(model, clip_list[clip_cnt])
                bboxes,labels,scores=analyse_result(result,score_thres)
                temp_cat,temp_score,temp_id,box_cnt,box_dict = get_inference_block_mask(bboxes,labels,scores,img_scale,box_cnt) # 每一个图块内处理
                mask_category[topleft_y:topleft_y+step,topleft_x:topleft_x+step]=temp_cat[margin:margin+step,margin:margin+step]    # 赋值中心区域
                mask_score[topleft_y:topleft_y+step,topleft_x:topleft_x+step]=temp_score[margin:margin+step,margin:margin+step]
                mask_id[topleft_y:topleft_y+step,topleft_x:topleft_x+step]=temp_id[margin:margin+step,margin:margin+step]
                dict_for_clips.update(box_dict)

            mask_category,mask_id=merge_clip_top_and_left((j,i),mask_category,mask_id,step,0.8) # 合并当前图块与左&上的边缘部分
            predict_png[topleft_y:topleft_y+step,topleft_x:topleft_x+step] = clip_list[clip_cnt][margin:margin+step,margin:margin+step] # 原图拼接
            clip_cnt+=1 # clip数量

    # FIXME:导出box结果(只是对每一个box求覆盖点最小和最大坐标)
    # [1,box_cnt] box-id
    dict_for_box={}
    id_list=list(Counter(mask_id.flatten()))
    # print(len(id_list))
    # print(id_list)
    for i in range(len(id_list)):
        id=id_list[i]
        if id==0:
            continue
        dict_for_box[id]=[w,h,-1,-1,dict_for_clips[id][4],dict_for_clips[id][5]] # x1,y1,x2,y2,category,score
    for j in range(h):
        for i in range(w):
            id=mask_id[j,i]
            if id==0:
                continue
            # x1,y1,x2,y2
            temp=dict_for_box[id]
            temp[0]=min(temp[0],i)
            temp[1]=min(temp[1],j)
            temp[2]=max(temp[2],i)
            temp[3]=max(temp[3],j)
            dict_for_box[id]=temp
    predict_png,final_box_list=label_vis(predict_png,id_list,dict_for_box)
    predict_png = predict_png[:image_h,:image_w]    # 去除右下边界
    mask_category=mask_category[:image_h,:image_w]*15
    # cv.imwrite("d:\\compound_img.jpg",predict_png)
    # cv.imwrite("d:\\compound_mask.jpg",mask_category)
    
    return final_box_list # predict_png,

def analyse_result(result,score_thr=0.5):
    """
    将model推理结果转化为可用数据，并根据score_thr筛选box
    """
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)

    labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
        # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)

    assert bboxes.shape[1] == 5
    scores = bboxes[:, -1]
    inds = scores > score_thr
    bboxes = bboxes[inds, :]
    labels = labels[inds]
    scores = bboxes[:, -1]

    return bboxes,labels,scores

def paint_chinese_opencv(im,chinese,position,fontsize,color):#opencv输出中文
    img_PIL = Image.fromarray(cv.cvtColor(im,cv.COLOR_BGR2RGB))# 图像从OpenCV格式转换成PIL格式
    font = ImageFont.truetype('simhei.ttf',fontsize,encoding="utf-8")
    #color = (255,0,0) # 字体颜色
    #position = (100,100)# 文字输出位置
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position,chinese,font=font,fill=color)# PIL图片上打印汉字 # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    img = cv.cvtColor(np.asarray(img_PIL),cv.COLOR_RGB2BGR)# PIL图片转cv2 图片
    return img

def label_vis(img,id_list,dict_box):
    bbox_color = (0,0,255)
    text_color = (255,0,0)
    width, height = img.shape[1], img.shape[0]
    font = cv.FONT_HERSHEY_SIMPLEX  # 使用默认字体
    cnt=len(id_list)
    box_list=[]
    for i in range(len(id_list)):
        id=id_list[i]
        if id==0:
            continue
        x1,y1,x2,y2,label,score=dict_box[id]
        print("({},{})   |   ({},{})  |  {}".format(x1,y1,x2,y2,MYCLASSES[label]))
        if x2-x1<=10 or y2-y1<=10:  # 过于细长的box跳过
            continue
        cv.rectangle(img, (x1,y1), (x2,y2), bbox_color, thickness=2)
        box_list.append([x1,y1,x2-x1,y2-y1,label,score])
        # cv.putText(img, MYCLASSES[label] + ': ' + str(score), (x1,y1), font, 0.6,text_color, 1)
        img = paint_chinese_opencv(img,MYCLASSES[label],(x1,y1),12,bbox_color)

    return img,box_list
        

if __name__ == "__main__":
    
    time_start=time.time()
    # Specify the path to model config and checkpoint file
    config_file = 'configs/yolox/yolox_s_8x8_300e_coco.py'
    checkpoint_file = 'work_dirs4/best_bbox_mAP_epoch_300.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    window_size=400
    center_size=300
    image_path="data/109.jpg"
    image = Image.open(image_path)
    image = np.asarray(image)
    img_size=(image.shape[0],image.shape[1])
    clip_list=get_clips(image,window_size,center_size)
    print("get clips done-----")
    clip_for_inference(model,clip_list,img_size,window_size,center_size)
    time_end=time.time()
    print('totally cost: {}'.format(time_end-time_start))
    # TODO:评估标准
    # FIXME:最佳window_size,center_size(400,300)
    # python dilatedInference.py
    # TODO:是否需要merge | 是否需要膨胀 | 
    # FIXME:是否需要减去width<10的box
