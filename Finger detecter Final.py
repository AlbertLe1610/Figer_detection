import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

def convert(img, color_space='bgr2hsv'):
    return cv2.cvtColor(img, eval('cv2.COLOR_%s' % color_space.upper()))

def thresholding(img, inrange: list):
    th = cv2.inRange(img,(inrange[0]),(inrange[1]))
    return th

def blur(img):
    th_blur = cv2.medianBlur(img,7)
    th_blur[th_blur > 0] = 255
    return th_blur

def floodFill(img):
    #Mask
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    # Floodfill from point (0, 0)
    floodFill = cv2.floodFill(img.copy(), mask, (0,0), 255)
    finalImg = 255 - floodFill[1] + img
    return finalImg

def cancelDistorsion(img):
    img_blur = blur(img)
    img_floodFill = floodFill(img_blur)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    erosion = cv2.erode(img_floodFill,kernel,iterations = 1)
    return erosion

def handIsolation(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_max = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    result = np.zeros_like(img)     
    cv2.drawContours(result, [cnt_max], -1, 1, -1) 
    return result

def cutHand(img):
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(70,70)) #70, 70
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel2)
    return tophat

def contours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def drawContours(img, contours):
    dc = cv2.drawContours(img, contours, -1, (0,255,0), 3)
    return dc

def Dilation(img):
   kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
   dilation = cv2.dilate(img,kernel,iterations = 1)
   return dilation

def Erosion(img):
  kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
  erode = cv2.erode(img, kernel, iterations = 1)
  return erode

def main(path):
    image = cv2.imread(path)
    # image_hsv = convert(image, color_space='BGR2HSV')
    # hsv_H = image_hsv[...,0]
    # H_thresholding = thresholding(hsv_H, [110, 180])
    image_ycrcb = convert(image, color_space='BGR2YCr_Cb')
    lower_ycrcb_values = np.array((80, 130, 75), dtype="uint8")
    upper_ycrcb_values = np.array((240, 170, 124), dtype="uint8")
    ycrcb_thresholding = thresholding(image_ycrcb, [lower_ycrcb_values, upper_ycrcb_values])
    ycrcb_thresholding = ~cv2.inRange(ycrcb_thresholding, (0), (0))
    dilation = Dilation(ycrcb_thresholding)
    thresholding_cd = cancelDistorsion(dilation)
    erosion = Erosion(thresholding_cd)
    handFinal = handIsolation(erosion)
    cd_resize = cv2.resize(handFinal, None, fx=435/handFinal.shape[1], fy=435/handFinal.shape[1])
    resize_cuthand = cutHand(cd_resize)
    cuthand_c1 = contours(resize_cuthand)
    c1_nd = drawContours(resize_cuthand, cuthand_c1)
    nd_c2 = contours(c1_nd)
    d = 0
    for c in nd_c2:
        if (cv2.contourArea(c))>1500:
            d+=1
            c = (c * image.shape[1]/435).astype(np.uint)
            _ = cv2.drawContours(image, [c], -1, (0,255,0), 3)
    print("Number of finger:", d)
    print("Image is at: D:\Python\Result")
    image = cv2.putText(image, "Number of finger: " + str(d), (40,40), cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 255, 0), 2, cv2.LINE_AA, bottomLeftOrigin=False) #image = cv2.putText(image, 'OpenCV', org, font,  fontScale, color, thickness, cv2.LINE_AA) 
    return image
    
def saveImg(img, path):
    cv2.imwrite(path, img)
if __name__ == "__main__":
    path = glob.glob("*.J*")
    for i in path:
        image = main(i)
        saveImg(image, './Result/%s' %i.split("/")[-1])







