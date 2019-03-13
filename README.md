# photometric-stereo
Implementation of simple photometric stereo like below picture.


<img src="./figure/figure1.png" width=90%>

## Environment
Ubuntu 18.04  
Python3.6(Anaconda3-5.2.0)



## Dependency

+ OpenCV3
+ numpy
+ sys
+ os
+ scipy





## How to run


### Dataset

Prepare proper images and masks for photometric stereo. In order to estimate light source direction, this code require chrome images that taken same lighting environment.  
  
Note that the images used in the example program are taken from the following project page: https://courses.cs.washington.edu/courses/cse455/10wi/projects/project4/psmImages_png.zip





### Usage

```
python main.py argvs[1] argvs[2] argvs[3]



argvs[1]  :  path to input folder   
-->  {abc}/{abc}.{number}.png, {abc}/{abc}.mask.png

argvs[2]  :  path to chrome folder   
-->  argvs[1] and argvs[2] have to taken same lighting condition

argvs[3]  :  number of image

Options:
windowsize: see mask2tiny function

```
