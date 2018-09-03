import cv2
import os, shutil
import numpy as np
import pdb
#Paths

main_dir="/media/ujwal/My files/Work/Non ML/Flowchart/"
input_dir=main_dir+"Input"
data_dir=main_dir+"Data"
output_dir=main_dir+"Output"

#Constants
kernel=np.ones((5,5),np.uint8)
upper=float("inf")
lower=4000
threshold=5
ArrowThresh=4000
window_size=300




def isreasonable(value):
	if value>lower and value< upper:
		return True
	else:
		return False	

def removeFolder(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): 
                shutil.rmtree(file_path)
        except Exception as e:
            print(e)

def createFolder(directory):
    try:
    
        os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory)


def load_dataset(folder):
    data = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            data.update({filename:img})#make dict with filename as key and pixel intensity matrix as value
    return data

def extract_templates(Data):
    template_contours={}
    for i in Data:
        
        im = Data[i]
        
        edge=cv2.Canny(im,5,100)
        edge=cv2.dilate(edge,kernel)
        edge=cv2.dilate(edge,kernel)
        edge=cv2.dilate(edge,kernel)
        edge=cv2.dilate(edge,kernel)
        
        _,contours,_= cv2.findContours(edge,2,1)

        cntl=contours[0]
        max=cv2.contourArea(contours[0])
        for cnt in contours:
            if cv2.contourArea(cnt)>max:
                max=cv2.contourArea(cnt)
                cntl=cnt

        template_contours.update({i:cntl})  #append Shapename:Contour Structure
    return template_contours    


def match_with_templates(Input,Templates):
    answer={}
    arrows={}
    for i in Input:
        inanswer=[]
        inarrows=[]
        im = Input[i]
        block_count=0;
        edge=cv2.Canny(im,5,100)
        edge=cv2.dilate(edge,kernel)
        edge=cv2.dilate(edge,kernel)
        # edge=cv2.erode(edge,kernel)
        
        # cv2.imshow('image',edge)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img1=np.zeros(im.shape)
        img2=np.zeros(im.shape)

        _,contours,_= cv2.findContours(edge,2,1)
        noCollide=True
        prev=float("inf")
        flag=1
        for cnt in contours:
            curr=np.mean(cnt,axis=0)
            if np.linalg.norm(curr-prev)>75:
                if flag:
                    print(curr,prev)
                    flag=0
                noCollide=True

            else:
            	noCollide=False

            if isreasonable(cv2.contourArea(cnt)):
                min=float("inf")
                jmin=''
                for j in Templates:
                	ret = cv2.matchShapes(cnt,Templates[j],1,0.0)
                	# print(ret,j)
                	if ret<min:
                		min=ret
                		jmin=j
                # print()		
                if min<threshold:
                    if noCollide:
                        block_count=block_count+1
                        inanswer.append(("block "+str(block_count),jmin,curr,cv2.contourArea(cnt)))
                        cv2.drawContours(img1, [cnt], 0, (0,255,0), 3)
                        # cv2.imshow('image',img1)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                    # else:
                    #     inarrows.append(cnt)
                    #     cv2.drawContours(img2, [cnt], 0, (0,255,0), 3)
            else:
                inarrows.append(cnt)
                cv2.drawContours(img2, [cnt], 0, (0,255,0), 3)
            prev=curr        
                    
                    
                    
        cv2.imwrite(os.path.join(output_dir,i[:-5]+' blocks.jpeg'),img1)
        cv2.imwrite(os.path.join(output_dir,i[:-5]+' arrows.jpeg'),img2)
        answer.update({i:inanswer})
        arrows.update({i:inarrows})
    return answer,arrows               

def checker(to_check,img):
    # img= cv2.copyMakeBorder(img,window_size,window_size,window_size,window_size,cv2.BORDER_CONSTANT,value=0)
    # cv2.imwrite('bordered.jpg',img)
    arrtailmatch=[]
    arrow_head_data=load_dataset(main_dir+"Arrows/arrowheads")
    # tail_data=load_dataset(main_dir+"Arrows/tails")
    arrow_head_templates=extract_templates(arrow_head_data)
    # tail_templates=extract_templates(tail_data)
    # return arrow_head_templates,tail_templates
    for i in to_check:
        comp=img[(i[1]-window_size//2):(i[1]+window_size//2),(i[0]-window_size//2):(i[0]+window_size//2)]
        # if(!comp):
        #     continue
        # print(comp.dtype)
        # cv2.imwrite('howeven.jpg',comp)
        # comp=cv2.imread('howeven.jpg',0)
        comp=np.uint8(comp)
        # print(comp.dtype)

        # edge=cv2.Canny(img,5,100)
        # edge=cv2.dilate(edge,kernel)
        # edge=cv2.dilate(edge,kernel)    
        # imgtemp=np.zeros(comp.shape)
        _,cnt,_=cv2.findContours(comp,2,1)
        retminarr=float("inf")
        # retmintail=float("inf")
        for c in cnt:
            # print(c)
            # cv2.drawContours(imgtemp,[c],0,255,3)
            # cv2.imshow('imgtemp',imgtemp)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            for j in arrow_head_templates:
                ret = cv2.matchShapes(c,arrow_head_templates[j],1,0.0)
                # print(str(ret)+"head")
                if ret<retminarr:
                    retminarr=ret
                    # print("taken")
            # for j in tail_templates:
            #     ret = cv2.matchShapes(c,tail_templates[j],1,0.0)
                # print(str(ret)+"tail")
                # if ret<retmintail:
                #     retmintail=ret
                    # print("taken")
        # print("appended:")
        # print(retminarr,retmintail)
        arrtailmatch.append(retminarr) 
        # pdb.set_trace()
    return arrtailmatch                  

def listify(x):
	return x.tolist()


def distance(a,b):
    return (a[0]-b[0])**2+(a[1]-b[1])**2






                        
def connection_finder(Input,block_list,arrows):
    # print(arrows)
    for i in block_list:
        imgo=Input[i]
        img=np.zeros((Input[i].shape[0],Input[i].shape[1]))
        img2=np.zeros((Input[i].shape[0],Input[i].shape[1],3))
        cv2.drawContours(img,arrows[i],-1,255,3)
        imgb=np.float32(img)
        dst=cv2.cornerHarris(imgb,2,3,0.21)
        # cv2.imwrite('Arrwork.jpg',img)            
        
        # pdb.set_trace()#cv2.imwrite('temp.jpg',dst)
        
        to_check=list(np.transpose(np.nonzero(dst)))
        to_check=list(map(listify,to_check))
        # print(to_check)
        arrtailmatch=checker(to_check,img)
        for j in arrows[i]:
            arrmin=float("inf")
            # tailmin=float("inf")
            arrcandidate=j[0][0].tolist()
            tailcandidate=None
            for point in j:
                point=point[0].tolist()
                if point in to_check:
                    index=to_check.index(point)
                    if arrtailmatch[index]<arrmin:
                        arrmin=arrtailmatch[index]
                        arrcandidate=point
                    # if arrtailmatch[index][1]<tailmin:
                    #     tailmin=arrtailmatch[index][1]
                    #     tailcandidate=point
            # if arrcandidate:
            img2[arrcandidate[1]][arrcandidate[0]]=(255,0,0) #head of arrow
            taildist=0
            flag=1
            for point in j:
                point=point[0].tolist()
                if flag:
                    print(point,arrcandidate)
                    flag=0
                dist=distance(point,arrcandidate)
                if dist>taildist:
                    taildist=dist
                    tailcandidate=point

            if tailcandidate:
                img2[tailcandidate[1]][tailcandidate[0]]=(0,255,0)
        img2=cv2.dilate(img2,kernel)
        img2=cv2.dilate(img2,kernel)
        img2=cv2.dilate(img2,kernel)
        cv2.imwrite('ArrowTail.jpg',img2)            






                







def main():
    # pdb.set_trace()
    Input=load_dataset(input_dir)
    Data=load_dataset(data_dir)

    Templates=extract_templates(Data)
    
    block_list,arrows=match_with_templates(Input,Templates)

    for i in block_list:
    	print()
    	print(i[0:-5]+':-')
    	for j in block_list[i]:
    		print(j[0]+' : '+str(j[1])+' @ '+str(j[2][0])+" with an area of "+str(j[3])+" sq units")	
    	print()	

    # pdb.set_trace()
    connection_finder(Input,block_list,arrows)



    # os.system("clear")








if __name__ == '__main__':
    main()

