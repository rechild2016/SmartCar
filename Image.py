import math
import numpy as np
import cv2

fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 0.8
fontcolor = (0, 250, 0)

def ImageProcess(image, type):
    _, im_gray = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)  # 这里进行了反转颜色

    img2 = cv2.Canny(im_gray, 100, 200)

    img3 = np.ones((60, 160)) * 255
    img4 = np.ones((60, 160)) * 255

    _, contours, hierarchy = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    c_max = []
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        print(len(contours[i]), area)
        # 处理掉小的轮廓区域，这个区域的大小自己定义。
        if (len(contours[i]) > 30 or area > 10):
            c_max.append(cnt)

    print("有效轮廓: ", len(c_max))

    cv2.drawContours(img3, c_max, -1, (0, 255, 0), 2)
    cv2.drawContours(img4, contours, -1, (0, 100, 0), 2)

    mid = midLine(c_max)
    for i in range(1, len(mid)):
        cv2.line(img4,mid[i-1],mid[i],(0,255,0),thickness=2)

    im1 = cv2.resize(im_gray, (200, 200))
    im2 = cv2.resize(img3, (200, 200))
    im3 = cv2.resize(img4, (200, 200))

    cv2.putText(im3, type, (60, 60), fontface, fontscale, fontcolor)
    return np.hstack((im1, im3, im2))

def Least_squares(x,y):
    x_ = x.mean()
    y_ = y.mean()
    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)
    for i in np.arange(len(x)):
        k = (x[i]-x_)* (y[i]-y_)
        m += k
        p = np.square( x[i]-x_ )
        n = n + p
    a = m/n
    b = y_ - a* x_
    return a,b


def midLine(contours):
    num = 0
    contourPoints  = [[] for n in range(60)]
    for contour in contours:
        contour_arr = np.array(contour).reshape(-1,2)
        for item in contour_arr:
            contourPoints[item[1]].append(item[0])

    mid = []
    for i in range(60):
        ave=0
        if len(contourPoints[59 -i]) > 1 and len(contourPoints[59 -i]) <= 6 :
            ave = math.ceil((max(contourPoints[59-i]) + min(contourPoints[59-i]))/2)
            if ( i >5 and math.fabs(ave - mid[-1][0]) > 10):
                ave = 2*mid[-1][0] -mid[-2][0]
            mid.append((ave, 59 -i))
        print(i," :",contourPoints[59 -i], ave)

    print(mid)
    return mid

# image = cv2.imread("./fig/Images/shizi/016.jpg")
# image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# im1 = ImageProcess(image,"shizi")
#
# cv2.imshow("img", im1)
# cv2.waitKey(0)