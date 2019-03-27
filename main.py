"""
 ----------------------------------------------------
  @Author: tsukasa
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-02-26 17:58:39
  @Last Modified by:   Tsukasa Nozawa
  @Last Modified time: 2019-03-27 13:44:35
 ----------------------------------------------------

  Usage:
   python main.py argvs[1] argvs[2] argvs[3]...
  
   argvs[1]  :  image folder path   -->   {abc}/{abc}.{number}.png, {abc}/{abc}.mask.png
   argvs[2]  :  chrome folder path   -->  argvs[1] and argvs[2] have to taken same lighting condition
   argvs[3]  :  number of image
 
  Options:
   windowsize: see mask2tiny function



"""

print(__doc__)


import numpy as np
import cv2
import sys
import os
import time

### my functions
import pms as ps
import obj_functions as ob




def load_data(SUBJECT, Ni):


  bgr = {}
  gray = {}
  mask = {}
  N = len(SUBJECT)

  for i in range(N):

    ps.progress_bar(i, N-1)
    s = SUBJECT[i]
    dirname = os.path.basename(os.path.dirname(s))
    bgr[dirname] = np.array([cv2.imread(s+'{0}.{1:d}.png'.format(dirname, x)) for x in range(Ni)])
    gray[dirname] = np.array([cv2.imread(s+'{0}.{1:d}.png'.format(dirname, x), cv2.IMREAD_GRAYSCALE) for x in range(Ni)])
    mask[dirname] = cv2.imread(s+'{0}.mask.png'.format(dirname), cv2.IMREAD_GRAYSCALE)


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
  start = time.time()


  ## prepare data set for estimation
  ## chrome is nessesarry to estimate light direction
  input_dirname = os.path.basename(os.path.dirname(input_path))
  chrome_dirname = os.path.basename(os.path.dirname(chrome_path))
  SUBJECT_path = [input_path, chrome_path]
  SUBJECT_list = [input_dirname, chrome_dirname]


  ## load all image
  print("Step1: load images from {0}".format(SUBJECT_path))
  bgr, gray, mask = load_data(SUBJECT_path, Ni)
  small_mask = mask2tiny(mask[SUBJECT_list[0]], 3)
  tiny_mask_path = os.path.join(outpath, SUBJECT_list[0]+"_tiny_mask.png")
  cv2.imwrite(tiny_mask_path, small_mask)
  print("\n")


  ########################
  ## 2. compute light
  ########################
  ## estimate light direction from chrome image
  lights = np.zeros((Ni, 3))
  print("Step2: Estimate light direction from chrome image")
  for i in range(Ni):
    ps.progress_bar(i, Ni-1)
    lights[i, :] = ps.comp_light(gray[SUBJECT_list[1]][i], mask[SUBJECT_list[1]])
  
  print("\n<result>\n -->\n {0}\n".format(lights))




  ########################
  ## 3. compute normal
  ########################
  print("Step3: Estimate normal")
  normal=ps.comp_normal(gray[SUBJECT_list[0]], small_mask, lights)
  normal_map = normal[:, :, ::-1]
  normalmap_path = os.path.join(outpath, SUBJECT_list[0]+"_normal.png")
  cv2.imwrite(normalmap_path, (normal_map+1.0)/2.0*255)
  print("\n")




  ########################
  ## 4. comp_albedo
  ########################
  print("Step4: Estimate albedo")
  albedo = ps.comp_albedo(bgr[SUBJECT_list[0]], small_mask, lights, normal)
  albedo_path = os.path.join(outpath, SUBJECT_list[0]+"_albedo.png")
  cv2.imwrite(albedo_path, albedo*255) 
  print("\n")



  ########################
  ## 5. comp_depth
  ########################
  print("Step5: Estimate depth from normal")
  depth = ps.comp_depth_4edge(small_mask, normal)
  depth_path = os.path.join(outpath, SUBJECT_list[0]+"_depth.png")
  cv2.imwrite(depth_path, (1.0 - (depth / np.max(depth))) * 255)
  print("done!!\n")
  end = time.time()



  ########################
  ## 6. save output
  ########################
  merage_result = os.path.join(outpath, SUBJECT_list[0]+"_results_merage.png")
  ply_result = os.path.join(outpath, SUBJECT_list[0]+"_recovered.ply")
  obj_result = os.path.join(outpath, SUBJECT_list[0]+"_recovered.obj")
  print("Step6: save result as\n --> \n{0}\n{1}\n{2}".format(merage_result, ply_result, obj_result))
  normal_image = np.array((normal_map+1.0)/2.0*255, dtype=np.uint8)
  albedo_image = np.array(albedo*255, dtype=np.uint8)
  depth_image = np.array((1.0 - (depth / np.max(depth))) * 255, dtype=np.uint8 )
  depth_image_rgb = cv2.cvtColor(depth_image, cv2.COLOR_GRAY2RGB)


  ### save result as image
  results = np.hstack((np.hstack((albedo_image, normal_image)), depth_image_rgb))
  cv2.imwrite(merage_result, np.array(results, dtype=np.uint8))
  # cv2.imshow("results", np.array(results, dtype=np.uint8))
  # cv2.waitKey(0)


  ### save result as 3d file
  vertex, triangle = ob.Depth2VerTri(depth, small_mask)
  ob.save_as_ply(ply_result, depth, normal, albedo, small_mask, triangle)
  ob.writeobj(obj_result, vertex, triangle)
  print("done!!\n")
  print("calculation time: {0}[sec]".format(round(end - start, 2)))




main()