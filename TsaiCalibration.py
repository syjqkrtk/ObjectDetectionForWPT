import numpy as np
from sympy import Symbol, solve, Eq

def Find_K_Matrix(refPt, world):
    world_x = world[0][0]
    world_y = world[0][1]
    world_z = world[0][2]
    imag_x = refPt[0][0]
    imag_y = refPt[0][1]
    A_temp = np.array(
        [[world_x, world_y, world_z, 1, 0, 0, 0, 0, -world_x * imag_x, -world_y * imag_x, -world_z * imag_x],
         [0, 0, 0, 0, world_x, world_y, world_z, 1, -world_x * imag_y, -world_y * imag_y, -world_z * imag_y]])

    B_temp = np.array([[imag_x], [imag_y]])

    A = A_temp
    B = B_temp
    for i in range(1,6):
        world_x = world[i][0]
        world_y = world[i][1]
        world_z = world[i][2]
        imag_x = refPt[i][0]
        imag_y = refPt[i][1]
        A_temp = np.array([[world_x, world_y, world_z, 1, 0, 0, 0, 0, -world_x*imag_x, -world_y*imag_x, -world_z*imag_x],
                 [0, 0, 0, 0, world_x, world_y, world_z, 1, -world_x*imag_y, -world_y*imag_y, -world_z*imag_y]])

        B_temp = np.array([[imag_x], [imag_y]])

        A = np.vstack([A, A_temp])
        B = np.vstack([B, B_temp])


    K_temp = np.matmul(np.linalg.pinv(A), B)
    K = np.vstack([K_temp, 1])
    K = K.reshape(3,4)

    return K



def Line_gen(imag_x, imag_y, K):
    y = Symbol('y')
    z = Symbol('z')
    
    dir1 = np.array([K[0][0]-K[2][0]*imag_x, K[0][1]-K[2][1]*imag_x, K[0][2]-K[2][2]*imag_x])
    dir2 = np.array([K[1][0]-K[2][0]*imag_y, K[1][1]-K[2][1]*imag_y, K[1][2]-K[2][2]*imag_y])
   
    dir_new = np.cross(dir1, dir2)
    
    eq1 = Eq(((K[0][1]-K[2][1]*imag_x)*y + (K[0][2]-K[2][2]*imag_x)*z + K[0][3]-K[2][3]*imag_x), 0)
    eq2 = Eq(((K[1][1]-K[2][1]*imag_y)*y + (K[1][2]-K[2][2]*imag_y)*z + K[1][3]-K[2][3]*imag_y), 0)
    
    sol_temp = solve([eq1, eq2], y, z)
    sol = np.hstack([0, sol_temp[y], sol_temp[z]])
        
    return dir_new, sol

def Est_Point(direction, point):
    t = Symbol('t')
    s = Symbol('s')
    
    eq1 = Eq((((direction[0][0]*t+point[0][0])-(direction[1][0]*s+point[1][0]))*direction[0][0] + ((direction[0][1]*t+point[0][1])-(direction[1][1]*s+point[1][1]))*direction[0][1] + ((direction[0][2]*t+point[0][2])-(direction[1][2]*s+point[1][2]))*direction[0][2]), 0)
    eq2 = Eq((((direction[0][0]*t+point[0][0])-(direction[1][0]*s+point[1][0]))*direction[1][0] + ((direction[0][1]*t+point[0][1])-(direction[1][1]*s+point[1][1]))*direction[1][1] + ((direction[0][2]*t+point[0][2])-(direction[1][2]*s+point[1][2]))*direction[1][2]), 0)
    
    y= solve([eq1, eq2], t, s)
    
    p1 = np.array([direction[0][0]*y[t]+point[0][0], direction[0][1]*y[t]+point[0][1], direction[0][2]*y[t]+point[0][2]])
    p2 = np.array([direction[1][0]*y[s]+point[1][0], direction[1][1]*y[s]+point[1][1], direction[1][2]*y[s]+point[1][2]])
    
    return (p1+p2)/2

def ImgTo3D(box1, box2, Area, RESOLUTION):
    if Area == 1:
        K1 = np.load('C:\\Users\\LOSTARK\\Dropbox\\Development\\Data\\Reference1.npy')
        K2 = np.load('C:\\Users\\LOSTARK\\Dropbox\\Development\\Data\\Reference2.npy')
        Image_x1 = np.floor(RESOLUTION[0]*(box1[0][1]+box1[0][3])/2)
        Image_y1 = np.floor(RESOLUTION[1]*(box1[0][0]+box1[0][2])/2)
        Image_x2 = np.floor(RESOLUTION[0]*(box2[0][1]+box2[0][3])/2)
        Image_y2 = np.floor(RESOLUTION[1]*(box2[0][0]+box2[0][2])/2)
    else:
        K1 = np.load('C:\\Users\\LOSTARK\\Dropbox\\Development\\Data\\Reference3.npy')
        K2 = np.load('C:\\Users\\LOSTARK\\Dropbox\\Development\\Data\\Reference4.npy')
        Image_x1 = np.floor(RESOLUTION[0]*(box1[0][1]+box1[0][3])/2)
        Image_y1 = np.floor(RESOLUTION[1]*(box1[0][0]+box1[0][2])/2)
        Image_x2 = np.floor(RESOLUTION[0]*(box2[0][1]+box2[0][3])/2)
        Image_y2 = np.floor(RESOLUTION[1]*(box2[0][0]+box2[0][2])/2)
    dir1, sol1 = Line_gen(Image_x1, Image_y1, K1)
    dir2, sol2 = Line_gen(Image_x2, Image_y2, K2)
    
    return Est_Point([dir1, dir2], [sol1, sol2])