"""
 ----------------------------------------------------
  @Author: tsukasa
  @Affiliation: Waseda University
  @Email: rinsa@suou.waseda.jp
  @Date: 2019-02-28 04:12:17
  @Last Modified by:   Tsukasa Nozawa
  @Last Modified time: 2019-02-28 05:14:20
 ----------------------------------------------------


"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import cv2




def comp_light(img, mask):
  '''
  1. find C(cx, cy, cz) and H(hx, hy, hz)
   --> C: center, H: heighlight position  

  2. Compute the surface normal at the point
   --> |H - C| = N

  3. Compute light source direction from normal
   --> L = 2(N*R)*N - R   (In this time R should have same vector with V.)
   --> R == V
   --> L = 2(N*V)*N -V

  '''

  ## 1. assume the viewer is at (0, 0, 1)
  view = np.array([0, 0, 1], dtype=np.float32)
  C = np.zeros((3,), dtype=np.float32)
  H = np.zeros((3,), dtype=np.float32)


  ## 2. Find the center and radius of the sphere from the mask
  ret, thresh = cv2.threshold(mask, 127, 255, 0)
  image, contours, hierarchy = cv2.findContours(thresh, 1, 2)
  cnt = contours[0]
  (cx, cy), cr = cv2.minEnclosingCircle(cnt)
  # C[0] = cx
  # C[1] = cy
  # print((cx, cy))
  # plt.imshow(cv2.circle(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (int(cx), int(cy)), int(cr), (0,255,0), 2))
  # plt.show()


  ## 3. Compute a weighted average of the brightest pixel locations.
  img[mask < 0.9*255] = 0
  ret, thresh = cv2.threshold(img, 250, 255, 0)
  image, contours, hierarchy = cv2.findContours(thresh, 1, 2)
  cnt = contours[0]
  M = cv2.moments(cnt)
  hx = M['m10']/M['m00']
  hy = M['m01']/M['m00']
  # H[0] = hx
  # H[1] = hy
  # print((int(hx), int(hy)))
  # plt.imshow(cv2.circle(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (int(hx), int(hy)), 3, (0, 255, 0), 1))
  # plt.show()


  ## 4. Compute the surface normal at the point
  cz = 0.0
  hz = np.sqrt(cr**2 - ((H[0]-C[0])**2 + (H[1]-C[1])**2))
  N = np.array([hx - cx, hy - cy, hz - cz], dtype=np.float32)
  n = N / np.linalg.norm(N)


  
  ## 5. Compute light source direction from normal
  lights = 2 * np.dot(n, view) * n - view
  lights = lights / np.linalg.norm(lights)

  return lights




def comp_normal(grays, mask, lights):
  '''
  I = L * n_tilde 
      note: n_tilde = dot(n, rho)
  
  --> L^T * I = L^T * L * n_tilde 
  --> n_tilda = (L^T * L)^(-1) * L^T * I

  Then n = n_tilda / |n_tilda|
  '''

  ## 1. prepare normal map array and mask array
  normal = np.zeros((mask.shape[0], mask.shape[1], 3)) 
  isAVaild = mask.copy().ravel() # size:(h*w, )
  

  ## 2. create each matrixs for pms
  I = grays.copy().reshape(grays.shape[0],grays.shape[1]*grays.shape[2]) # size:(Ni, h*w)
  # Lt = lights.T # size: (3, Ni)
  # LtL_inv = np.linalg.inv(np.dot(Lt, lights)) # size: (3, 3)
  # n_tilde = np.dot(LtL_inv, np.dot(Lt, I)) # size: (3, Ni)
  L_inv = np.linalg.pinv(lights) # size: (3, 3)
  

  ## 3. solve pms
  n_tilde = np.dot(L_inv, I) # size: (3, Ni)
  n_tilde_t = n_tilde.T # size:(Ni, 3)

  
  ## 4. create normal map
  for i in range(n_tilde_t.shape[0]):
    
    ## albedo for gray image 
    rho = np.linalg.norm(n_tilde_t[i])
    
    ## current position
    pixel_y = i//mask.shape[1]
    pixel_x = i%mask.shape[1]


    if(isAVaild[i] == 0 or rho == 0):
      normal[pixel_y, pixel_x, :] = np.array([0,0,0])
    
    else:
      n =  n_tilde_t[i] / rho
      n = n / np.linalg.norm(n)
      normal[pixel_y, pixel_x, :] = n
  

  return normal




def comp_albedo(bgrs, mask, lights, normal):
  '''
  I_r = rho_r * L * N
  I_g = rho_g * L * N
  I_b = rho_b * L * N

  --> 

  rho_x = I_X * (L * N)^(-1)

  '''

  ## 1. prepare albedo map array and each matirixes to calculate albedo
  albedo = np.zeros((mask.shape[0], mask.shape[1], 3))
  N = normal.copy().reshape(normal.shape[0]*normal.shape[1],normal.shape[2]) # size: (h*w, 3)
  LN = np.dot(lights, N.T) # size: (Ni, h*w)


  ## 2. calculate each posiiton's albedo
  for i in range(bgrs.shape[1]):
    for j in range(bgrs.shape[2]):

      ## current position
      pixel = (i * mask.shape[1]) + j
      
      ## each matrix
      LN_pixel = LN[:, pixel].reshape(bgrs.shape[0], 1) # size: (Ni, 1)
      LN_pixel_inv = np.linalg.pinv(LN_pixel) # size: (1, Ni)

      ## albedo(r, g, b) for position(i, j)
      albedo[i, j, :] = np.dot(LN_pixel_inv, bgrs[:, i, j, :])
  

  ## 3. normalize
  albedo = (albedo - np.min(albedo)) / (np.max(albedo) - np.min(albedo))



  return albedo




def comp_depth(mask, normal):

  '''
  "arbitrary point p(x, y, Z(x, y)) and Np = (nx, ny, nz)"

  v1 = (x+1, y, Z(x+1, y)) - p
     = (1, 0, Z(x+1, y) - Z(x, y))

  Then, dot(Np, v1) == 0 #right
  0 = Np * v1
    = (nx, ny, nz) * (1, 0, Z(x+1,y) - Z(x, y))
    = nx + nz(Z(x+1,y) - Z(x, y))

  --> Z(x+1,y) - Z(x, y) = -nx/nz = p

  Also dot(Np, v2) is same #up
  --> Z(x,y+1) - Z(x, y) = -ny/nz = q

  
  Finally, apply least square to find Z(x, y).
  A: round matrix
  x: matrix of Z(x, y)
  b: matrix of p and q 

  A*x = b


  (--> might be left bottom as well???)

  '''


  ## 1. prepare matrix for least square
  A = sp.lil_matrix((mask.size * 2, mask.size))
  b = np.zeros(A.shape[0], dtype=np.float32)


  ## 2. set normal
  nx = normal[:,:,0].ravel()
  ny = normal[:,:,1].ravel()
  nz = normal[:,:,2].ravel()
  

  ## 3. fill b 
  ##  --> 0~nx.shape[0] is for v1
  ##  --> .... v2, v3, v4
  b[0:nx.shape[0]] = -nx/(nz+1e-8)
  b[nx.shape[0]:b.shape[0]] = -ny/(nz+1e-8)
  


  ## 4. fill A 
  dif= mask.size
  w = mask.shape[1]
  h = mask.shape[0]
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):

      ## current pixel om matrix
      pixel = (i * w) + j
      

      ## for v1(right)
      if j != w-1:
        A[pixel, pixel]   = -1
        A[pixel, pixel+1] = 1

      ## for v2(up)
      if i != h-1:
        A[pixel+dif, pixel]   = -1
        A[pixel+dif, pixel+w] = 1
  


  ## 5. solve Ax = b
  AtA = A.transpose().dot(A)
  Atb = A.transpose().dot(b)
  x, info = sp.linalg.cg(AtA, Atb)
  


  ## 6. create output matrix
  depth -= np.min(depth)
  depth[mask == 0] = 0.0


  return depth



def comp_depth_4edge(mask, normal):

  '''
  "arbitrary point p(x, y, Z(x, y)) and Np = (nx, ny, nz)"

  v1 = (x+1, y, Z(x+1, y)) - p
     = (1, 0, Z(x+1, y) - Z(x, y))

  Then, dot(Np, v1) == 0 #right
  0 = Np * v1
    = (nx, ny, nz) * (1, 0, Z(x+1,y) - Z(x, y))
    = nx + nz(Z(x+1,y) - Z(x, y))

  --> Z(x+1,y) - Z(x, y) = -nx/nz = p

  Also dot(Np, v2) is same #up
  --> Z(x,y+1) - Z(x, y) = -ny/nz = q

  
  Finally, apply least square to find Z(x, y).
  A: round matrix
  x: matrix of Z(x, y)
  b: matrix of p and q 

  A*x = b


  (--> might be left bottom as well???)

  '''


  ## 1. prepare matrix for least square
  A = sp.lil_matrix((mask.size * 4, mask.size))
  b = np.zeros(A.shape[0], dtype=np.float32)


  ## 2. set normal
  nx = normal[:,:,0].ravel()
  ny = normal[:,:,1].ravel()
  nz = normal[:,:,2].ravel()


  ## 3. fill b 
  ##  --> 0~nx.shape[0] is for v1
  ##  --> .... v2, v3, v4
  b[0:nx.shape[0]]               = -nx/(nz+1e-8)
  b[nx.shape[0]:2*nx.shape[0]]   = -ny/(nz+1e-8)
  b[2*nx.shape[0]:3*nx.shape[0]] = -nx/(nz+1e-8)
  b[3*nx.shape[0]:b.shape[0]]    = -ny/(nz+1e-8)
  

  ## 4. fill A 
  dif= mask.size
  w = mask.shape[1]
  h = mask.shape[0]
  for i in range(mask.shape[0]):
    for j in range(mask.shape[1]):

      ## current pixel om matrix
      pixel = (i * w) + j
      

      ## for v1(right)
      if j != w-1:
        A[pixel, pixel]   = -1
        A[pixel, pixel+1] = 1

      ## for v2(up)
      if i != h-1:
        A[pixel+dif, pixel]   = -1
        A[pixel+dif, pixel+w] = 1
  

      ## for v3(left)
      if j != 0:
        A[pixel+(2*dif), pixel]   = 1
        A[pixel+(2*dif), pixel-1] = -1

      ## for v4(bottom)
      if i != 0:
        A[pixel+(3*dif), pixel]   = 1
        A[pixel+(3*dif), pixel-w] = -1


  ## 5. solve Ax = b
  AtA = A.transpose().dot(A)
  Atb = A.transpose().dot(b)
  x, info = sp.linalg.cg(AtA, Atb)
  

  ## 6. create output matrix
  depth = x.reshape(mask.shape)
  depth -= np.min(depth)
  depth[mask == 0] = 0.0

  return depth