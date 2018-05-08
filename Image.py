import math
import numpy as np
import cv2
from scipy import signal

fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 0.8
fontcolor = (0, 250, 0)

def ImageProcess(image, type):
    _, im_gray = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)  # 这里进行了反转颜色

    img2 = cv2.Canny(im_gray, 100, 200)

    img3 = np.ones((60, 160)) * 255
    img4 = np.ones((60, 160)) * 255
    img5 = np.ones((60, 160)) * 255
    _, contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contoursFliter = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        # print(len(contours[i]), area)
        # 处理掉小的轮廓区域，这个区域的大小自己定义。
        if (len(contours[i]) > 30 or area > 10):
            contoursFliter.append(cnt)

    print("有效轮廓: ", len(contoursFliter))

    cv2.drawContours(img3, contoursFliter, -1, (0, 255, 0), 2)
    cv2.drawContours(img4, contours, -1, (0, 100, 0), 2)

    mid = midLine(contoursFliter)
    for i in range(1, len(mid)):
        cv2.line(img4,mid[i-1],mid[i],(0,255,0),thickness=2)

    im1 = cv2.resize(im_gray, (160, 120))
    im2 = cv2.resize(img3, (160, 120))
    im3 = cv2.resize(img4, (160, 120))

    cv2.putText(im2, type, (50, 90), fontface, fontscale, fontcolor)
    return np.hstack((im1, im2, im3))

# def Least_squares(x,y):
#     x_ = x.mean()
#     y_ = y.mean()
#     m = np.zeros(1)
#     n = np.zeros(1)
#     k = np.zeros(1)
#     p = np.zeros(1)
#     for i in np.arange(len(x)):
#         k = (x[i]-x_)* (y[i]-y_)
#         m += k
#         p = np.square( x[i]-x_ )
#         n = n + p
#     a = m/n
#     b = y_ - a* x_
#     return a,b

def midLine(contours):
    num = 0
    contourPoints  = [[] for n in range(60)]
    for contour in contours:
        contour_arr = np.array(contour).reshape(-1,2)
        for item in contour_arr:
            contourPoints[item[1]].append(item[0])

    mid = []
    mid.append((80,61))
    mid.append((80, 60))
    for i in range(60):
        ave = -1
        if len(contourPoints[59 -i]) > 1 and len(contourPoints[59 -i]) <= 6 :
            ave = math.ceil((max(contourPoints[59-i]) + min(contourPoints[59-i]))/2)
            if ( i >5 and math.fabs(ave - mid[-1][0]) > 10):
                ave = 2*mid[-1][0] -mid[-2][0]
            mid.append((ave, 59 -i))
        # print(i," :",contourPoints[59 -i], ave)

    # print(mid)
    return mid

if __name__ == '__main__':
    image = cv2.imread("./fig/Images/shizi/063.jpg")
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    im1 = ImageProcess(image,"shizi")

    cv2.imshow("img", im1)
    cv2.waitKey(0)