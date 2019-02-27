"""
 ----------------------------------------------------
  @Author: tsukasa
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-02-26 17:58:39
  @Last Modified by:   Tsukasa Nozawa
  @Last Modified time: 2019-02-28 05:14:12
 ----------------------------------------------------

  Usage:
   python main.py argvs[1] argvs[2] argvs[3]...
  
   argvs[1]  :  image folder paht   -->   {abc}/{abc}.{number}.png, {abc}/{abc}.mask.png
   argvs[2]  :  chrome folder path   -->  argvs[1] and argvs[2] have to taken same lighting condition
   argvs[3]  :  number of image
 
  Options:
   windowsize: see mask2tiny function



"""


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import cv2
import sys
import os
import pms as ps
import obj_functions as ob




def load_data(SUBJECT, Ni):


  print(SUBJECT)

  bgr = {}
  gray = {}
  mask = {}

  for s in SUBJECT:
    dirname = os.path.basename(os.path.dirname(s))
    bgr[dirname] = np.array([cv2.imread(s+'/{0}.{1:d}.png'.format(dirname, x)) for x in range(Ni)])
    gray[dirname] = np.array([cv2.imread(s+'/{0}.{1:d}.png'.format(dirname, x), cv2.IMREAD_GRAYSCALE) for x in range(Ni)])
    mask[dirname] = cv2.imread(s+'/{0}.mask.png'.format(dirname), cv2.IMREAD_GRAYSCALE)


  return bgr, gray, mask






def mask2tiny(mask, window):

  '''
  naive approach to remove noise around border
  '''

  # mask
  mask = np.array(mask, dtype=np.uint8)
  eroded = cv2.erode(mask, np.ones((int(window), int(window)), np.uint8)) # 0~1

  return eroded








def main():
  ########################
  ## 1. set config
  ########################
  argvs = sys.argv
  input_path = argvs[1]
  chrome_path = argvs[2]
  outpath = input_path
  Ni = int(argvs[3])


  ## prepare data set for estimation
  ## chrome is nessesarry to estimate light direction
  input_dirname = os.path.basename(os.path.dirname(input_path))
  chrome_dirname = os.path.basename(os.path.dirname(chrome_path))
  SUBJECT_path = [input_path, chrome_path]
  SUBJECT_list = [input_dirname, chrome_dirname]


  ## load all image
  bgr, gray, mask = load_data(SUBJECT_path, Ni)
  small_mask = mask2tiny(mask[SUBJECT_list[0]], 3)
  cv2.imwrite("{0}/{1}_tiny_mask.png".format(outpath, SUBJECT_list[0]), small_mask)



  ########################
  ## 2. compute light
  ########################
  ## estimate light direction from chrome image
  lights = np.zeros((Ni, 3))
  for i in range(Ni):
    lights[i, :] = ps.comp_light(gray[SUBJECT_list[1]][i], mask[SUBJECT_list[1]])
    print("Estimated light source No.{0} --> {1}".format(i, lights[i, :]))




  ########################
  ## 3. compute normal
  ########################
  normal=ps.comp_normal(gray[SUBJECT_list[0]], small_mask, lights)
  normal_map = normal[:, :, ::-1]
  cv2.imwrite("{0}/{1}_normal.png".format(outpath, SUBJECT_list[0]), (normal_map+1.0)/2.0*255)





  ########################
  ## 4. comp_albedo
  ########################
  albedo = ps.comp_albedo(bgr[SUBJECT_list[0]], small_mask, lights, normal)
  cv2.imwrite("{0}/{1}_albedo.png".format(outpath, SUBJECT_list[0]), albedo*255) 




  ########################
  ## 5. comp_depth
  ########################
  depth = ps.comp_depth_4edge(small_mask, normal)
  cv2.imwrite("{0}/{1}_depth.png".format(outpath, SUBJECT_list[0]), (1.0 - (depth / np.max(depth))) * 255)




  ########################
  ## 6. save output
  ########################
  normal_image = np.array((normal_map+1.0)/2.0*255, dtype=np.uint8)
  albedo_image = np.array(albedo*255, dtype=np.uint8)
  depth_image = np.array((1.0 - (depth / np.max(depth))) * 255, dtype=np.uint8 )
  depth_image_rgb = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)

  results = np.hstack((np.hstack((normal_image, albedo_image)), depth_image_rgb))
  cv2.imwrite("{0}/{1}_results_merage.png".format(outpath, SUBJECT_list[0]), np.array(results, dtype=np.uint8))
  cv2.imshow("results", np.array(results, dtype=np.uint8))
  cv2.waitKey(0)
  # plt.imshow(np.flip(results, 2))
  # plt.show()


  vertex, triangle = ob.Depth2VerTri(depth, small_mask)
  ob.save_as_ply("{0}/{1}_recovered.ply".format(outpath, SUBJECT_list[0]), depth, normal, albedo, small_mask, triangle)
  ob.writeobj("{0}/{1}_recovered.obj".format(outpath, SUBJECT_list[0]), vertex, triangle)





main()