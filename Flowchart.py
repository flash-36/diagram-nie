import cv2
import os, shutil
import numpy as np

#Paths

main_dir="/media/ujwal/My files/Work/Non ML/Flowchart/"
input_dir=main_dir+"Input"
data_dir=main_dir+"Data"

#Constants
kernel=np.ones((5,5),np.uint8)
upper=float("inf")
lower=4000
threshold=5





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

        template_contours.update({i[0:-4]:cntl})  #append Shapename:Contour Structure
    return template_contours    


def match_with_templates(Input,Templates):
    answer={}
    for i in Input:
        inanswer=[]
        im = Input[i]
        
        edge=cv2.Canny(im,5,100)
        edge=cv2.dilate(edge,kernel)
        edge=cv2.dilate(edge,kernel)
        # edge=cv2.erode(edge,kernel)
        
        # cv2.imshow('image',edge)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        img=np.zeros(im.shape)
        _,contours,_= cv2.findContours(edge,2,1)
        noCollide=True
        prev=float("inf")
        for cnt in contours:
        	curr=np.mean(cnt,axis=0)
        	if np.linalg.norm(curr-prev)>75:
        		noCollide=True
        	else:
        		noCollide=False

        	if isreasonable(cv2.contourArea(cnt)) and noCollide:
        		min=float("inf")
        		jmin=''
        		for j in Templates:
        			ret = cv2.matchShapes(cnt,Templates[j],1,0.0)
        			print(ret,j)
        			if ret<min:
        				min=ret
        				jmin=j
        		print()		
        		if min<threshold:
        			inanswer.append((jmin,curr,cv2.contourArea(cnt)))
        			prev=curr
        			cv2.drawContours(img, [cnt], 0, (0,255,0), 3)
        			# cv2.imshow('image',img)
        			# cv2.waitKey(0)
        			# cv2.destroyAllWindows()
        			
        cv2.imwrite('image.jpeg',img)
        answer.update({i:inanswer})
    return answer    			

        				







def main():
    Input=load_dataset(input_dir)
    Data=load_dataset(data_dir)

    Templates=extract_templates(Data)
    
    Final_list=match_with_templates(Input,Templates)

    for i in Final_list:
    	print()
    	print(i[0:-5]+':-')
    	for j in Final_list[i]:
    		print(str(j[0])+' @ '+str(j[1][0])+" with an area of "+str(j[2])+" sq units")	
    	print()	






    # os.system("clear")








if __name__ == '__main__':
    main()


