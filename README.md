# photometric-stereo
Implementation of simple photometric stereo like below picture.

<div align="center">
  <img src="./figure/figure1.png" width=100%>
</div>

## Environment
Ubuntu 18.04  
Python3.6



## Dependency

+ OpenCV4
+ numpy
+ sys
+ os
+ scipy
+ argparse




## Usage


### Dataset

Prepare proper images and masks for photometric stereo. In order to estimate light source direction, this code require chrome images that taken same lighting environment.  
  
Note that the images used in the example program are taken from the following project page: https://courses.cs.washington.edu/courses/cse455/10wi/projects/project4/psmImages_png.zip





### How to run

```
python main.py


optional arguments:
  -h, --help        show this help message and exit
  -i , --input      path to input images folder --> {abc}/{abc}.{number}.png,
                    {abc}/{abc}.mask.png
  -c , --chrome     path to chrome images folder --> same lighting condition with input
                    images
  -n , --numimage   number of images for estimation
  -w , --window     window size to make mask small

```

## References

[1] [Woodham, Robert J](https://www.cs.ubc.ca/~woodham/). "Photometric method for determining surface orientation from multiple images." Optical engineering 19.1 (1980): 191139.
[[Paper](https://www.cs.ubc.ca/~woodham/papers/Woodham80c.pdf "Paper")]  
