"""
 ----------------------------------------------------
  @Author: tsukasa
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-02-26 17:58:39
 ----------------------------------------------------

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
import argparse




def load_data(SUBJECT, Ni):


  bgr = {}
  gray = {}
  mask = {}
  N = len(SUBJECT)
  print(SUBJECT)

  for i in range(N):

    # ps.progress_bar(i, N-1)
    s = SUBJECT[i]
    dirname = os.path.basename(os.path.dirname(s))
    print(dirname)
    bgr[dirname] = np.array([cv2.imread(os.path.join(s,'{0}.{1:d}.png'.format(dirname, x))) for x in range(Ni)])
    gray[dirname] = np.array([cv2.imread(os.path.join(s,'{0}.{1:d}.png'.format(dirname, x)), cv2.IMREAD_GRAYSCALE) for x in range(Ni)])
    mask[dirname] = cv2.imread(os.path.join(s,'{0}.mask.png'.format(dirname)), cv2.IMREAD_GRAYSCALE)


  return bgr, gray, mask






def mask2tiny(mask, window):

  '''
  naive approach to remove noise around border
  '''

  # mask
  mask = np.array(mask, dtype=np.uint8)
  eroded = cv2.erode(mask, np.ones((int(window), int(window)), np.uint8)) # 0~1

  return eroded





def parse_arguments():

  #### ----- set arguments
  parser = argparse.ArgumentParser(description="photometric stereo")
  parser.add_argument("-i",
                      "--input",
                      type=str,
                      default="./example/rock/",
                      metavar='',
                      help='path to input images folder\n --> {abc}/{abc}.{number}.png, {abc}/{abc}.mask.png')
  parser.add_argument("-c",
                      "--chrome",
                      type=str,
                      default="./example/chrome/",
                      metavar='',
                      help='path to chrome images folder\n --> same lighting condition with input images')
  parser.add_argument("-n",
                      "--numimage",
                      type=int,
                      default=9,
                      metavar='',
                      help="number of images for estimation")
  parser.add_argument("-w",
                      "--window",
                      type=int,
                      default=3,
                      metavar='',
                      help="window size to make mask small")
  args = parser.parse_args()
  args.outdir = os.path.dirname(args.input)


  #### ----- print arguments
  text  = "\n<input arguments>\n"
  for key in vars(args):
    text += "{}: {}\n".format(key.ljust(15), str(getattr(args, key)).ljust(30))
  print(text)

  return args








def main():
  ########################
  ## 1. set config
  ########################
  args = parse_arguments()
  outpath = args.input
  # args.numimage = int(argvs[3])
  start = time.time()


  ## prepare data set for estimation
  ## chrome is nessesarry to estimate light direction
  input_dirname = os.path.basename(os.path.dirname(args.input))
  chrome_dirname = os.path.basename(os.path.dirname(args.chrome))
  SUBJECT_path = [args.input, args.chrome]
  SUBJECT_list = [input_dirname, chrome_dirname]


  ## load all image
  print("Step1: load images from {0}".format(SUBJECT_path))
  bgr, gray, mask = load_data(SUBJECT_path, args.numimage)
  small_mask = mask2tiny(mask[SUBJECT_list[0]], args.window)
  tiny_mask_path = os.path.join(outpath, SUBJECT_list[0]+"_tiny_mask.png")
  cv2.imwrite(tiny_mask_path, small_mask)
  print("\n")


  ########################
  ## 2. compute light
  ########################
  ## estimate light direction from chrome image
  lights = np.zeros((args.numimage, 3))
  print("Step2: Estimate light direction from chrome image")
  for i in range(args.numimage):
    ps.progress_bar(i, args.numimage-1)
    lights[i, :] = ps.comp_light(gray[SUBJECT_list[1]][i], mask[SUBJECT_list[1]])
  
  print("\n<result>\n -->\n {0}\n".format(lights))




  ########################
  ## 3. compute normal
  ########################
  print("Step3: Estimate normal")
  normal=ps.comp_normal(gray[SUBJECT_list[0]], small_mask, lights)
  normal_map = normal[:, :, ::-1]
  normal_map[small_mask == 0] = [1.0, 1.0, 1.0]
  normalmap_path = os.path.join(outpath, SUBJECT_list[0]+"_normal.png")
  cv2.imwrite(normalmap_path, (normal_map+1.0)/2.0*255)
  print("\n")




  ########################
  ## 4. comp_albedo
  ########################
  print("Step4: Estimate albedo")
  albedo = ps.comp_albedo(bgr[SUBJECT_list[0]], small_mask, lights, normal)
  albedo[small_mask == 0] = [1.0, 1.0, 1.0]
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
  results = np.hstack((albedo_image, normal_image, depth_image_rgb))
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