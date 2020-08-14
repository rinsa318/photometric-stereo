"""
 ----------------------------------------------------
  @Author: tsukasa
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-02-28 02:09:19
 ----------------------------------------------------


"""




import numpy as np
import sys
import os



#################
# create triangles from point cloud(grid)
#################


def gen_triangle(mask):


  ## 1. prepare index for triangle
  index_map = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
  index = 0
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):

      if(mask[i, j] == 0):
        continue

      index_map[i, j] = index
      index += 1



  ## 2. create flags for triangle
  mask_bool = np.array(mask, dtype = 'bool')


  ## flag1
  left_up = mask_bool.copy()
  left_up[1:mask.shape[0],:] *= mask_bool[0:mask.shape[0]-1,:] # multiply up movement
  left_up[:,1:mask.shape[1]] *= mask_bool[:,0:mask.shape[1]-1] # multiply left movement
  left_up[0,:] = False
  left_up[:,0] = False

  ## flag2
  right_down = mask_bool.copy()
  right_down[0:mask.shape[0]-1,:] *= mask_bool[1:mask.shape[0],:] # multiply down movement
  right_down[:,0:mask.shape[1]-1] *= mask_bool[:,1:mask.shape[1]] # multiply right movement
  right_down[mask.shape[0]-1,:] = False
  right_down[:,mask.shape[1]-1] = False


  '''
    (i-1, j-1) -----(i-1, j) ------(i-1, j+1)
        |              |               |
        |              |               |
        |              |               |
     (i, j-1) ------ (i, j) ------ (i, j+1)
        |              |               |
        |              |               |
        |              |               |
    (i+1, j-1) ----(i+1, j)------(i+1, j+1)

  flag1 means: Δ{ (i, j), (i-1, j), (i, j-1) }
  flag1 means: Δ{ (i, j), (i+1, j), (i, j+1) }

  otherwise:
    case1: is not locate on edge(i, j ==0) and exist left up point
    --> Δ{ (i, j), (i-1, j-1), (i, j-1) }
    
    case2: is not locate on edge(i, j ==shape-1) and exist right down
    --> Δ{ (i, j), (i+1, j+1), (i, j+1) }
    
  '''


  ## 3. fill triangle list like above
  triangle = []
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):

      ## outside --> ignore
      if(not(mask_bool[i, j])):
        continue

      ## flag1
      if(left_up[i, j]):
        triangle.append( [index_map[i, j], index_map[i-1, j], index_map[i, j-1]] )

      ## flag2
      if(right_down[i, j]):
        triangle.append( [index_map[i, j], index_map[i+1, j], index_map[i, j+1]] )

      ## otherwise
      if(not(left_up[i, j]) and not(right_down[i, j])):

        ## case1
        if(i != 0 and j != 0 and mask_bool[i, j-1] and mask_bool[i-1, j-1]):
          triangle.append( [index_map[i, j], index_map[i-1, j-1], index_map[i, j-1]] )

        ## case2
        if(i != mask_bool.shape[0]-1 and j != mask_bool.shape[1]-1 and mask_bool[i, j+1] and mask_bool[i+1, j+1]):
          triangle.append( [index_map[i, j], index_map[i+1, j+1], index_map[i, j+1]] )


  return np.array(triangle, dtype=np.int64)




def Depth2VerTri(depth, mask):



  ## 1. prepare index for triangle and vertex array
  index_map = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int64)
  index = 0
  vertex = []

  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):

      if(mask[i, j] == 0):
        continue

      index_map[i, j] = index
      index += 1


      vertex.append( [i, j, depth[i, j]] )

  ## 2. create flags for triangle
  mask_bool = np.array(mask, dtype = 'bool')


  ## flag1
  left_up = mask_bool.copy()
  left_up[1:mask.shape[0],:] *= mask_bool[0:mask.shape[0]-1,:] # multiply up movement
  left_up[:,1:mask.shape[1]] *= mask_bool[:,0:mask.shape[1]-1] # multiply left movement
  left_up[0,:] = False
  left_up[:,0] = False

  ## flag2
  right_down = mask_bool.copy()
  right_down[0:mask.shape[0]-1,:] *= mask_bool[1:mask.shape[0],:] # multiply down movement
  right_down[:,0:mask.shape[1]-1] *= mask_bool[:,1:mask.shape[1]] # multiply right movement
  right_down[mask.shape[0]-1,:] = False
  right_down[:,mask.shape[1]-1] = False


  '''
    (i-1, j-1) -----(i-1, j) ------(i-1, j+1)
        |              |               |
        |              |               |
        |              |               |
     (i, j-1) ------ (i, j) ------ (i, j+1)
        |              |               |
        |              |               |
        |              |               |
    (i+1, j-1) ----(i+1, j)------(i+1, j+1)

  flag1 means: Δ{ (i, j), (i-1, j), (i, j-1) }
  flag1 means: Δ{ (i, j), (i+1, j), (i, j+1) }

  otherwise:
    case1: not on edge(i, j ==0) and exist left up point
    --> Δ{ (i, j), (i-1, j-1), (i, j-1) }
    
    case2: not on edge(i, j ==shape-1) and exist right down
    --> Δ{ (i, j), (i+1, j+1), (i, j+1) }
    
  '''


  ## 3. fill triangle list like above
  triangle = []
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):

      ## outside --> ignore
      if(not(mask_bool[i, j])):
        continue

      ## flag1
      if(left_up[i, j]):
        triangle.append( [index_map[i, j], index_map[i-1, j], index_map[i, j-1]] )

      ## flag2
      if(right_down[i, j]):
        triangle.append( [index_map[i, j], index_map[i+1, j], index_map[i, j+1]] )

      ## otherwise
      if(not(left_up[i, j]) and not(right_down[i, j])):

        ## case1
        if(i != 0 and j != 0 and mask_bool[i, j-1] and mask_bool[i-1, j-1]):
          triangle.append( [index_map[i, j], index_map[i-1, j-1], index_map[i, j-1]] )

        ## case2
        if(i != mask_bool.shape[0]-1 and j != mask_bool.shape[1]-1 and mask_bool[i, j+1] and mask_bool[i+1, j+1]):
          triangle.append( [index_map[i, j], index_map[i+1, j+1], index_map[i, j+1]] )


  return np.array(vertex, dtype=np.float32), np.array(triangle, dtype=np.int64)




#################
# for .obj
#################


def loadobj(path):
  vertices = []
  #texcoords = []
  triangles = []
  normals = []

  with open(path, 'r') as f:
    for line in f:
      if line[0] == '#':
        continue

      pieces = line.split(' ')

      if pieces[0] == 'v':
        vertices.append([float(x) for x in pieces[1:4]])      
      # elif pieces[0] == 'vt':
      #   texcoords.append([float(x) for x in pieces[1:]])
      elif pieces[0] == 'f':
        if pieces[1] == '':
            triangles.append([int(x.split('/')[0]) - 1 for x in pieces[2:]])
        else: 
            triangles.append([int(x.split('/')[0]) - 1 for x in pieces[1:]])
      elif pieces[0] == 'vn':
        normals.append([float(x) for x in pieces[1:]])
      else:
        pass

  return (np.array(vertices, dtype=np.float32),
            #np.array(texcoords, dtype=np.float32),
            np.array(triangles, dtype=np.int32))#,
            # np.array(normals, dtype=np.float32))




def writeobj(filepath, vertices, triangles):
  with open(filepath, "w") as f:
    for i in range(vertices.shape[0]):
      f.write("v {} {} {}\n".format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
    for i in range(triangles.shape[0]):
      f.write("f {} {} {}\n".format(triangles[i, 0] + 1, triangles[i, 1] + 1, triangles[i, 2] + 1))






def writeobj_with_uv(filepath, vertices, triangles, uv):
  basename = os.path.basename(filepath)
  with open(filepath, "w") as f:
    f.write("mtllib " + basename + "_material.mtl\n")
    
    for i in range(vertices.shape[0]):
      f.write("v {} {} {}\n".format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
    for i in range(uv.shape[0]):
      f.write("vt {} {}\n".format(uv[i, 0], uv[i, 1]))

    f.write("usemtl tex\n")
    # f.write("s off\n")
    for i in range(triangles.shape[0]):
      f.write("f {0}/{0} {1}/{1} {2}/{2}\n".format(triangles[i, 0] + 1, triangles[i, 1] + 1, triangles[i, 2] + 1))

  with open(filepath + "_material.mtl", "w") as f:
    # f.write("newmtl emerald\n")
    # f.write("Ns 600\n")
    # f.write("d 1\n")
    # f.write("Ni 0.001\n")
    # f.write("illum 2\n")
    # f.write("Ka 0.0215  0.1745   0.0215\n")
    # f.write("Kd 0.07568 0.61424  0.07568\n")
    # f.write("Ks 0.633   0.727811 0.633\n")
    # f.write("map_Kd sand.jpg\n")

    f.write("newmtl tex\n")
    f.write("Ns 96.078431\n")
    f.write("Ka 1.000000 1.000000 1.000000\n")
    f.write("Kd 0.640000 0.640000 0.640000\n")
    f.write("Ks 0.500000 0.500000 0.500000\n")
    f.write("Ke 0.000000 0.000000 0.000000\n")
    f.write("Ni 1.000000\n")
    f.write("d 1.000000\n")
    f.write("map_Kd sand.jpg\n")





def writepoint(filepath, vertices):
  with open(filepath, "w") as f:
    for i in range(vertices.shape[0]):
      f.write("v {} {} {}\n".format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
    # for i in range(triangles.shape[0]):
    #   f.write("f {} {} {}\n".format(triangles[i, 0] + 1, triangles[i, 1] + 1, triangles[i, 2] + 1))






def writeoff(filepath, vertices, triangles):
  with open(filepath, "w") as f:
    f.write("OFF\n")
    f.write("# convert by tsukasa\n")
    f.write("\n")
    f.write("{} {} {}\n".format(vertices.shape[0], triangles.shape[0], 0))
    for i in range(vertices.shape[0]):
      f.write("{} {} {}\n".format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))
    for i in range(triangles.shape[0]):
      f.write("3 {} {} {}\n".format(triangles[i, 0], triangles[i, 1], triangles[i, 2]))











#################
# for .ply
#################

# def readply():


def save_as_ply(filename, depth, normal, albedo, mask, triangle):

  mask_bool = np.array(mask, dtype = 'bool')
  Np = np.count_nonzero(mask)
  bgr = np.array(albedo * 255, dtype=np.uint8)
  # bgr = np.array(albedo, dtype=np.uint8)



  ## write ply file
  with open(filename, 'w') as fp:

    ## header infomation
    fp.write('ply\n')
    fp.write('format ascii 1.0\n')
    fp.write('element vertex {0}\n'.format(Np))
    fp.write('property float x\n')
    fp.write('property float y\n')
    fp.write('property float z\n')
    fp.write('property float nx\n')
    fp.write('property float ny\n')
    fp.write('property float nz\n')
    fp.write('property uchar blue\n')
    fp.write('property uchar green\n')
    fp.write('property uchar red\n')
    fp.write('element face {0}\n'.format(len(triangle)))
    fp.write('property list uchar int vertex_indices\n')
    fp.write('end_header\n')



    for i in range(depth.shape[0]):
      for j in range(depth.shape[1]):
          
        if(not(mask_bool[i, j])):
            continue

        fp.write('{0:e} {1:e} {2:e} {3:e} {4:e} {5:e} {6:d} {7:d} {8:d}\n'.format(i, j, depth[i, j], normal[i, j, 0], normal[i, j, 1], normal[i, j, 2], bgr[i, j, 0], bgr[i, j, 1], bgr[i, j, 2]))
    
    for i in range(len(triangle)):
      fp.write('3 {0:d} {1:d} {2:d}\n'.format(triangle[i][0], triangle[i][1], triangle[i][2]))



