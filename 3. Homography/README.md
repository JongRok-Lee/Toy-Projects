# Homography (DLT + RANSAC, PROSAC)
#### 변환할 이미지 평면
![](sources/02.jpg)
#### 변환 결과가 될 타겟 이미지 평면
![](sources/01.jpg)
### DLT (Direct Linear Transform)
$x_1 = Hx_2$ --> $x_1-Hx_2 = 0$ --> $||Ap|| = 0$  
$p = [H_{1} H_{2} \ldots H_{9}]$  
$A = UDV^{T}$, and $p = V[:, -1]$

### RANSAC, PROSAC 결과
#### OpenCV (Ground Truth)
![](results/opencv.png)
#### RANSAC (Method 1)
![](results/RANSAC.png)
#### PROSAC (Method 2)
![](results/PROSAC.png)
