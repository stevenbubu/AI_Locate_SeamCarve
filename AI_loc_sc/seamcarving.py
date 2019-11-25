# USAGE:
# python seam_carving.py (-resize | -remove) -im IM -out OUT [-mask MASK]
#                        [-rmask RMASK] [-dy DY] [-dx DX] [-vis] [-hremove] [-backward_energy]
# Examples:
# python seam_carving.py -resize -im demos/ratatouille.jpg -out ratatouille_resize.jpg 
#        -mask demos/ratatouille_mask.jpg -dy 20 -dx -200 -vis
# python seam_carving.py -remove -im demos/eiffel.jpg -out eiffel_remove.jpg 
#        -rmask demos/eiffel_mask.jpg -vis

import numpy as np
import cv2
import argparse
from numba import jit, cuda
from scipy import ndimage as ndi

import os
import time, shutil, random, math
import csv
import logging.config
import utils.log as log
logging.config.dictConfig(log.config)
logger = logging.getLogger("StreamLogger")
from config.seamcarving_config import FLAGS
import warnings
warnings.filterwarnings('ignore')
import threading

os.environ["CUDA_VISIBLE_DEVICES"]="0"    # select gpu device to use
ENERGY_MASK_CONST = 100000.0              # large energy value for protective masking
MASK_THRESHOLD = 10                       # minimum pixel intensity for binary mask
USE_FORWARD_ENERGY = True                 # if True, use forward energy algorithm
SEAM_COLOR = np.array([255, 200, 200])    # seam visualization color (BGR)
SEAM_COLOR_BLUE = np.array([255, 0, 0])   # seam visualization color (BGR)
SEAM_COLOR_RED = np.array([0, 0, 255])   # seam visualization color (BGR)
########################################
# UTILITY CODE
########################################

def backup(path, bkpath=FLAGS.backupfilepath, folders=[]):
    if not os.path.exists(bkpath):
        os.makedirs(bkpath)
    logger.info("> Backup file already exist.")

    usefd = filterUseFolders(path, folders)  
    usefdbkp = filterUseFolders(bkpath, folders)
    for f in usefd:
        fbp = f+str(2) if f in usefdbkp else f
        shutil.copytree(os.path.join(path, f), os.path.join(bkpath, fbp)) 
    [shutil.rmtree(os.path.join(bkpath, f)) for f in usefdbkp]

def belist(threads=2, folders=[], refers=[]):
    data = list()
    if len(refers) != 0:
        for k in range(len(folders)):
            for v in range(len(refers)):
                data.append((folders[k], refers[v]))
    else:
        data = folders
    
    array = []
    for idx in range(math.ceil(len(data)/threads)):
        # tmp = []
        if len(data) > threads:
            array.append(data[idx*threads:idx*threads+threads])
        else:
            array.append(data[idx*threads:len(data)])
    return array

def createFolder(folders, path=None):
    for folder in folders:
        if path is not None:
            fdpath = os.path.join(path, folder)    
        else:
            fdpath = folder
        try:
            if not os.path.exists(fdpath):
                os.makedirs(fdpath)
            else:
                logger.info('Folder already exists: ' +  folder)
        except OSError:
             logger.info('Error: Creating directory. ' + fdpath)

def filterUseFolders(path, folders=[]):
    files = os.listdir(path)
    files_tag = []
    [files_tag.append(f) for f in files for d in folders if str(d) in f]
    return files_tag

def save_csv(csv_path, csv_content):
    with open(csv_path, 'a') as csvfile:
        wr = csv.writer(csvfile)
        for i in range(len(csv_content)):
            wr.writerow(csv_content[i])

def read_txt(txt_path):
    with open(txt_path, 'r') as txtfile:
        line = txtfile.readlines()
    return line

def save_txt(txt_path, txt_content, string=""):
    with open(txt_path, 'a') as txtfile:
        txtfile.write(string + txt_content)    

def timing(sec):
    day = int(sec//(60*60*24));  day_rm = sec%(60*60*24)
    hour = int(day_rm//(60*60)); hour_rm = day_rm%(60*60)
    mins = int(hour_rm//60);     sec = hour_rm%60

    day = str("0"+str(day)) if day<10 else str(day)
    hour = str("0"+str(hour)) if hour<10 else str(hour)
    mins = str("0"+str(mins)) if mins<10 else str(mins)
    sec = str("0"+str(sec)) if sec<10 else str("%.2f"%sec)
    string = str(day) + ":" + str(hour) + ":" + str(mins) + ":" + str(sec)
    logger.info("Time: " + string)
    return string

def visualize(im, boolmask=None, rotate=False):
    vis = im.astype(np.uint8)
    if boolmask is not None:
        vis[np.where(boolmask == False)] = SEAM_COLOR
    if rotate:
        vis = rotate_image(vis, False)
    cv2.imshow("visualization", vis)
    cv2.waitKey(1)
    return vis

def resize(image, width):
    dim = None
    h, w = image.shape[:2]
    dim = (width, int(h * width / float(w)))
    return cv2.resize(image, dim)

def rotate_image(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)    

########################################
# ENERGY FUNCTIONS
########################################

def backward_energy(im):
    """
    Simple gradient magnitude energy map.
    """
    xgrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=1, mode='wrap')
    ygrad = ndi.convolve1d(im, np.array([1, 0, -1]), axis=0, mode='wrap')
    
    grad_mag = np.sqrt(np.sum(xgrad**2, axis=2) + np.sum(ygrad**2, axis=2))

    # vis = visualize(grad_mag)
    # cv2.imwrite("backward_energy_demo.jpg", vis)
    
    return grad_mag

@jit
def forward_energy(im):
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.

    Vectorized code adapted from
    https://github.com/axu2/improved-seam-carving.
    """
    h, w = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))
    
    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)
    
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU
    
    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        
        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)
    
    # vis = visualize(energy)
    # cv2.imwrite("forward_energy_demo.jpg", vis)     
        
    return energy

########################################
# SEAM HELPER FUNCTIONS
######################################## 

@jit
def add_seam(im, seam_idx):
    """
    Add a vertical seam to a 3-channel color image at the indices provided 
    by averaging the pixels values to the left and right of the seam.

    Code adapted from https://github.com/vivianhylee/seam-carving.
    """
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1, 3))
    for row in range(h):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                p = np.average(im[row, col: col + 2, ch])
                output[row, col, ch] = im[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]
            else:
                p = np.average(im[row, col - 1: col + 1, ch])
                output[row, : col, ch] = im[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = im[row, col:, ch]

    return output

@jit
def add_seam_grayscale(im, seam_idx, seammap=False):
    """
    Add a vertical seam to a grayscale image at the indices provided 
    by averaging the pixels values to the left and right of the seam.
    """    
    h, w = im.shape[:2]
    output = np.zeros((h, w + 1)) 
    for row in range(h):
        col = seam_idx[row]
        if col == 0:
            p = 1 if seammap else np.average(im[row, col: col + 2])
            output[row, col] = im[row, col]
            output[row, col + 1] = p
            output[row, col + 1:] = im[row, col:]
        else:
            p = 1 if seammap else np.average(im[row, col - 1: col + 1])
            output[row, :col] = im[row, :col]
            output[row, col] = p
            output[row, col + 1:] = im[row, col:]

    return output

@jit
def remove_seam(im, boolmask):
    h, w = im.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return im[boolmask3c].reshape((h, w - 1, 3))

@jit
def remove_seam_grayscale(im, boolmask):
    h, w = im.shape[:2]
    return im[boolmask].reshape((h, w - 1))

@jit
def get_minimum_seam(im, mask=None, remove_mask=None):
    """
    DP algorithm for finding the seam of minimum energy. Code adapted from 
    https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
    """
    h, w = im.shape[:2]
    energyfn = forward_energy if USE_FORWARD_ENERGY else backward_energy
    M = energyfn(im)

    if mask is not None:
        M[np.where(mask > MASK_THRESHOLD)] = ENERGY_MASK_CONST

    # give removal mask priority over protective mask by using larger negative value
    if remove_mask is not None:
        M[np.where(remove_mask > MASK_THRESHOLD)] = ENERGY_MASK_CONST * 100

    backtrack = np.zeros_like(M, dtype=np.int)

    # populate DP matrix
    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(M[i-1, j:j+2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx+j]
            else:
                idx = np.argmin(M[i-1, j-1:j+2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i-1, idx + j-1]

            M[i, j] += min_energy

    # backtrack to find path
    seam_idx = []
    boolmask = np.ones((h, w), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in range(h-1, -1, -1):
        boolmask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()
    return np.array(seam_idx), boolmask

########################################
# MAIN ALGORITHM
######################################## 

def seams_removal(im, num_remove, mask=None, vis=False, rot=False):
    for _ in range(num_remove):
        seam_idx, boolmask = get_minimum_seam(im, mask)
        if vis:
            visualize(im, boolmask, rotate=rot)
        im = remove_seam(im, boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)
    return im, mask

def seams_insertion(im, num_add, mask=None, vis=False, rot=False):
    seams_record = []
    temp_im = im.copy()
    temp_mask = mask.copy() if mask is not None else None

    for _ in range(num_add):
        seam_idx, boolmask = get_minimum_seam(temp_im, temp_mask)
        if vis:
            visualize(temp_im, boolmask, rotate=rot)

        seams_record.append(seam_idx)
        temp_im = remove_seam(temp_im, boolmask)
        if temp_mask is not None:
            temp_mask = remove_seam_grayscale(temp_mask, boolmask)

    seams_record.reverse()

    for _ in range(num_add):
        seam = seams_record.pop()
        im = add_seam(im, seam)
        if vis:
            visualize(im, rotate=rot)
        if mask is not None:
            mask = add_seam_grayscale(mask, seam)

        # update the remaining seam indices
        for remaining_seam in seams_record:
            remaining_seam[np.where(remaining_seam >= seam)] += 2         

    return im, mask

########################################
# MAIN DRIVER FUNCTIONS
########################################

def seam_carve(im, dy, dx, mask=None, vis=False):
    im = im.astype(np.float64)
    h, w = im.shape[:2]
    assert h + dy > 0 and w + dx > 0 and dy <= h and dx <= w

    if mask is not None:
        mask = mask.astype(np.float64)

    output = im

    if dx < 0:
        output, mask = seams_removal(output, -dx, mask, vis)

    elif dx > 0:
        output, mask = seams_insertion(output, dx, mask, vis)

    if dy < 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = seams_removal(output, -dy, mask, vis, rot=True)
        output = rotate_image(output, False)

    elif dy > 0:
        output = rotate_image(output, True)
        if mask is not None:
            mask = rotate_image(mask, True)
        output, mask = seams_insertion(output, dy, mask, vis, rot=True)
        output = rotate_image(output, False)

    return output


def object_removal(im, rmask, mask=None, vis=False, horizontal_removal=False):
    im = im.astype(np.float64)
    rmask = rmask.astype(np.float64)
    if mask is not None:
        mask = mask.astype(np.float64)
    output = im

    h, w = im.shape[:2]

    if horizontal_removal:
        output = rotate_image(output, True)
        rmask = rotate_image(rmask, True)
        if mask is not None:
            mask = rotate_image(mask, True)

    while len(np.where(rmask > FLAGS.MASK_THRESHOLD)[0]) > 0:
        seam_idx, boolmask = get_minimum_seam(output, mask, rmask)
        if vis:
            visualize(output, boolmask, rotate=horizontal_removal)            
        output = remove_seam(output, boolmask)
        rmask = remove_seam_grayscale(rmask, boolmask)
        if mask is not None:
            mask = remove_seam_grayscale(mask, boolmask)

    num_add = (h if horizontal_removal else w) - output.shape[1]
    output, mask = seams_insertion(output, num_add, mask, vis, rot=horizontal_removal)
    if horizontal_removal:
        output = rotate_image(output, False)

    return output        


def save2Tiff(path, usdfile=[]):
    outfd = os.path.join(os.getcwd(), os.path.dirname(path).split("/")[-1] + "_tiff")
    createFolder([outfd])
    index = 0
    for f in usdfile:
        index += 1
        img_sor_path = os.path.join(path, f)
        img_des_path = os.path.join(outfd, f.split(".")[0] + ".tiff")
        img = cv2.imread(img_sor_path, cv2.IMREAD_UNCHANGED)
        cv2.imwrite(img_des_path, img)
        logger.info("Folder:" + os.path.dirname(path).split("/")[-1] + ", [" + str(index) + "] image:" + f + " save tiff process finish.")
    logger.info("Images save to tiff finish. Folders:" + os.path.dirname(path).split("/")[-1] + " Len:" + str(index))


def genMask(path, size, usdfd=[], masks=[]):
    maskfd = []
    [maskfd.append("mask" + str(cp)) for cp in masks]
    for folder in usdfd:
        basepath = os.path.join(path, folder)
        maskfdpath = []
        maskfdpath = [os.path.join(basepath + '_' + f) for f in maskfd]
        [createFolder([f]) for f in  maskfdpath]
        index = 0
        for idx in masks:
            img_des_path = maskfdpath[masks.index(idx)]     
            front = size//2; end = idx-int(size//2)
            block = int((end-front)//size)
            for j in range(block):
                T = front+j*size; B = T+size
                for i in range(block):
                    L = front+i*size; R = L+size 
                    index += 1     
                    mask = np.ones((idx,idx), dtype=np.uint8)*255
                    cv2.rectangle(mask, (L, T), (R, B), (0, 0), -1) 
                    filepath = os.path.join(img_des_path, "mask" + str(i+j*block) + ".tiff")
                    cv2.imwrite(filepath, mask)
                    logger.info("Folder:" + folder + ", mask [" + str(idx) + "], block [" + str(i+j*block) + "] process finish.")
            logger.info("Folder:" + folder + ", mask [" + str(idx) + "] process finish.")
    logger.info("Generate masks finish. Folders:" + usdfd[0] + " Len:" + str(len(usdfd)))    


def cropImg(path, usdfd=[], crops=[]):
    cropfd = []
    [cropfd.append("crop" + str(cp)) for cp in crops]
    for folder in usdfd:
        basepath = os.path.join(path, folder)
        cropfdpath = []
        cropfdpath = [os.path.join(basepath + '_' + f) for f in cropfd]
        [createFolder([f]) for f in  cropfdpath]
        index = 0
        for f in os.listdir(basepath):
            if(f.split(".")[-1] == "tiff"):
                index += 1
                img_sor_path = os.path.join(basepath, f)
                img = cv2.imread(img_sor_path)            
                for idx in crops:
                    img_des_path = os.path.join(cropfdpath[crops.index(idx)], f)
                    h, w = img.shape[:2]
                    h, w = (int(h * idx / float(w)), idx) if h > w else (idx, int(w * idx / float(h))) if w > h else (idx, idx)
                    img_resize = cv2.resize(img, (w, h))
                    img_crop = img_resize[(int(h//2)-int(idx/2)):(int(h//2) + int(idx/2)),:,:] if h> w else \
                                img_resize[:,(int(w//2)-int(idx/2)):(int(w//2) + int(idx/2)),:] if w > h else img_resize
                    cv2.imwrite(img_des_path, img_crop)
                    logger.info("Folder:" + folder + ", [" + str(index) + "] image:" + f + " crop:" + str(idx) + " process finish.")
        logger.info("Folder:" + folder + " process finish.")
    logger.info("Crop images process finish. Len:" + str(len(usdfd)))    


def findSCLoc(rmask):
    h, w = rmask.shape
    for j in range(h):
        for i in range(w):
            if rmask[j][i] == 0:
                center = (i,j)
    return center


def target_removal(im, num_remove, rmask=None, vis=False):
    rmask = rmask.astype(np.float64)
    seams_record = []

    for i in range(num_remove):
        # vertical seam remove
        seam_idx, boolmask = get_minimum_seam(im, remove_mask=rmask)
        if vis:
            visualize(im, boolmask, rotate=False)
        seams_record.append(seam_idx)
        im = remove_seam(im, boolmask)
        rmask = remove_seam_grayscale(rmask, boolmask)

        # horizontal seam remove
        im = rotate_image(im, True)
        rmask = rotate_image(rmask, True)
        seam_idx, boolmask = get_minimum_seam(im, remove_mask=rmask)
        if vis:
            visualize(im, boolmask, rotate=True)
        seams_record.append(seam_idx)
        im = remove_seam(im, boolmask)
        rmask = remove_seam_grayscale(rmask, boolmask)
        im = rotate_image(im, False)
        rmask = rotate_image(rmask, False)
        if i == num_remove-1:
            SCcenter = findSCLoc(rmask)


    # generate seam map
    h, w = im.shape[:2]
    seam_map = np.zeros((h, w))

    for _ in range(num_remove):    
        # horizontal seam add
        seam = seams_record.pop()
        seam_map = rotate_image(seam_map, True)
        seam_map = add_seam_grayscale(seam_map, seam, seammap=True)    

        # vertical seam add
        seam = seams_record.pop()
        seam_map = rotate_image(seam_map, False)
        seam_map = add_seam_grayscale(seam_map, seam, seammap=True)

    return im, seam_map, SCcenter


def ImgwithSeam(image, seammap, color=SEAM_COLOR_BLUE):
    h, w = seammap.shape
    for j in range(h):
        for i in range(w):
            if seammap[i][j] == 0:
                image[i,j,:] = color
    return image


def recordSCLoc(path, scimg, center, framesize=[]):
    h, w, _ = scimg.shape
    filename = path.split("/")[-1]
    # header = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    for fr in framesize:  
        xmin=0; ymin=0; xmax=0; ymax=0     
        xmin=center[0]-int(fr//2); xmax=center[0]+int(fr//2)
        ymin=center[1]-int(fr//2); ymax=center[1]+int(fr//2)
        labelrow = filename + " " + str(w) + " " + str(h) + " " + "Target" + " " + \
                    str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax)
        txt_path = path[:-5] + "_f" + str(fr) + ".txt"
        if os.path.exists(txt_path):
            os.remove(txt_path)
            save_txt(txt_path, labelrow)
        else:
            save_txt(txt_path, labelrow)
        logger.info("Image:" + filename + ", frame:" + str(fr) + " record location process finish.")
        cv2.waitKey(2)  # close window when a key press is detected


def frameSCLoc(path, scimg, center, framesize=[]):
    for fr in framesize:  
        L=0; T=0; R=0; B=0     
        L=center[0]-int(fr//2); R=center[0]+int(fr//2)
        T=center[1]-int(fr//2); B=center[1]+int(fr//2)
        scimg_frame = cv2.rectangle(scimg, (L, T), (R, B), (0, 0, 255), 1)
        cv2.imwrite(path, scimg_frame)
        logger.info("Image:" + path.split("/")[-1] + " frame process finish.")


def carveImg(path, size, carves=[], framesize=[]):
    usdfd = []
    usdfd = filterUseFolders(path, folders=carves)

    for carv in carves:
        start_carv_time = time.time()
        cropfd = [f for f in usdfd if str('crop') in f if str(carv) in f]
        maskfd = [f for f in usdfd if str('mask') in f if str(carv) in f]
        basepath = os.path.join(path, cropfd[0])
        maskpath = os.path.join(path, maskfd[0])
        seamcarvepath = basepath + "_seamcarve"
        seamcarveframepath = basepath + "_seamcarve_frame"
        seammappath = basepath + "_seammap"
        seammapimgpath = basepath + "_seammap_img"
        createFolder([seamcarvepath, seamcarveframepath, seammappath, seammapimgpath])
        files = os.listdir(basepath)
        maskfiles = os.listdir(maskpath)
        num = int(len(files)//len(maskfiles)); rem = int(len(files)%len(maskfiles))
        mask = []
        for i in range(len(maskfiles)):
            N = num+1 if i < rem else num
            file = random.sample(files, N) if files else None
            mask.append(file)
            [[files.remove(f) for f in file] if file else None]
            
        for i in range(len(maskfiles)):
            if mask[i]:
                start_mask_time = time.time()
                index = 0
                for f in mask[i]:
                    seamcarvepath_tmp = os.path.join(seamcarvepath, "mask" + str(i))
                    seammappath_tmp = os.path.join(seammappath, "mask" + str(i))
                    seammapimgpath_tmp = os.path.join(seammapimgpath, "mask" + str(i))
                    seamcarveframepath_tmp = os.path.join(seamcarveframepath, "mask" + str(i))
                    createFolder([seamcarvepath_tmp, seammappath_tmp, seammapimgpath_tmp, seamcarveframepath_tmp])
                    index += 1
                    img = cv2.imread(os.path.join(basepath, f), cv2.IMREAD_UNCHANGED)
                    rmaskfile = [f for f in maskfiles if "mask"+str(i)+".tiff" in f]
                    rmask = cv2.imread(os.path.join(maskpath, rmaskfile[0]), cv2.IMREAD_UNCHANGED)
                    img_output, seam_map, SCcenter = target_removal(img, size, rmask=rmask, vis=False)
                    recordSCLoc(os.path.join(seamcarvepath_tmp, f), scimg=img_output, center=SCcenter, framesize=framesize)
                    frameSCLoc(os.path.join(seamcarveframepath_tmp, f), scimg=img_output, center=SCcenter, framesize=[8])
                    seam_map = 255 - cv2.normalize(seam_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    seam_map_img = ImgwithSeam(img, seam_map, color=SEAM_COLOR_BLUE)
                    cv2.imwrite(os.path.join(seamcarvepath_tmp, f), img_output)
                    cv2.imwrite(os.path.join(seammappath_tmp, f), seam_map)
                    cv2.imwrite(os.path.join(seammapimgpath_tmp, f), seam_map_img)
                    logger.info("Folder:" + cropfd[0] + ", Mask:" + str(i) + ", [" + str(index) + "] Image:" + f + " seacmcarving process finish.")
                end_mask_time = time.time()
                mask_spend = timing(end_mask_time - start_mask_time)
                save_txt(path+'/time.txt', mask_spend, string="\nsize: "+str(carv)+" mask: "+str(i)+" seamcarve: ")
        end_carv_time = time.time()
        carv_spend = timing(end_carv_time - start_carv_time)
        save_txt(path+'/time.txt', carv_spend, string="\nsize: "+str(carv)+" seamcarve total: ")
        logger.info("Folder:" + cropfd[0] + ", Mask Folder:" + maskfd[0] + " seacmcarving process finish. Len:" + str(len(files)))
    logger.info("Seam Carve process finish.")

def trainData(path, trainpath, size, frame, num):
    usdfd = []
    usdfd = filterUseFolders(path, folders=[str(size)])
    usdscfd = []
    [usdscfd.append(f) for f in usdfd if "seamcarve" in f if "frame" not in f]
    scfdpath = os.path.join(path, usdscfd[0])
    trainfdpath = os.path.join(trainpath, "Train")
    testfdpath = os.path.join(trainpath, "Test")
    createFolder([trainfdpath, testfdpath])
    imgfiles = []; txtfiles = []
    [imgfiles.append(f) for f in os.listdir(scfdpath) if f.split(".")[-1] == "tiff"]
    [txtfiles.append(f) for f in os.listdir(scfdpath) if f.split(".")[-1] == "txt" if str(frame) in f]
    trainfiles=[]; traintxts=[]; testfiles=[]; testtxts=[]
    train_idx = random.sample(range(len(imgfiles)), k=num)
    [trainfiles.append(imgfiles[idx]) for idx in train_idx] 
    [testfiles.append(f) for f in imgfiles if f not in trainfiles]
    [traintxts.append(d) for f in trainfiles for d in txtfiles if f.split(".")[0] in d]
    [testtxts.append(f) for f in txtfiles if f not in traintxts]

    # Create train_labels.csv
    train_csv_des_path = os.path.join(trainfdpath, 'train_labels.csv')
    test_csv_des_path = os.path.join(testfdpath, 'test_labels.csv')
    csvholder = []
    header = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    csvholder.append(header)
    save_csv(train_csv_des_path, csvholder)
    save_csv(test_csv_des_path, csvholder)
    for f in trainfiles:
        img_sor_path = os.path.join(scfdpath, f)
        img_des_path = os.path.join(trainfdpath, f)
        shutil.copy(img_sor_path, img_des_path)
    for f in traintxts:
        csvholder = []
        txt_sor_path = os.path.join(scfdpath, f)
        row = read_txt(txt_sor_path)
        csvholder.append(row[0].split())
        save_csv(train_csv_des_path, csvholder)
    for f in testfiles:
        img_sor_path = os.path.join(scfdpath, f)
        img_des_path = os.path.join(testfdpath, f)
        shutil.copy(img_sor_path, img_des_path)
    for f in testtxts:
        csvholder = []
        txt_sor_path = os.path.join(scfdpath, f)
        row = read_txt(txt_sor_path)
        csvholder.append(row[0].split())
        save_csv(test_csv_des_path, csvholder)
    logger.info("Train and test data collect process finish.")

 
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group()
    group.add_argument("-resize", action='store_true')
    group.add_argument("-remove", action='store_true')

    ap.add_argument("-im", help="Path to image", default=FLAGS.infilepath)
    ap.add_argument("-out", help="Output file name", default=FLAGS.outfilepath)
    ap.add_argument("-mask", help="Path to (protective) mask")
    ap.add_argument("-rmask", help="Path to removal mask")
    ap.add_argument("-dy", help="Number of vertical seams to add/subtract(+/-)", type=int, default=FLAGS.dy)
    ap.add_argument("-dx", help="Number of horizontal seams to add/subtract(+/-)", type=int, default=FLAGS.dx)
    ap.add_argument("-vis", help="Visualize the seam removal process", action='store_true', default=True)
    ap.add_argument("-hremove", help="Remove horizontal seams for object removal", action='store_true')
    ap.add_argument("-backward_energy", help="Use backward energy map (default is forward)", action='store_true', default=True)
    args = vars(ap.parse_args())

    IMG_PATH, MASK_PATH, IMG_OUT_PATH, R_MASK_PATH = args["im"], args["mask"], args["out"], args["rmask"]

    start_time = time.time()
    outfddir = os.path.dirname(os.path.abspath(IMG_OUT_PATH)) if IMG_PATH.split(".")[-1].lower() in FLAGS.image_format \
                else os.path.abspath(IMG_OUT_PATH)

    if IMG_PATH.split(".")[-1].lower() in FLAGS.image_format:
        createFolder([outfddir])
        img = cv2.imread(IMG_PATH)
        assert img is not None
        mask = cv2.imread(MASK_PATH, 0) if MASK_PATH else None
        rmask = cv2.imread(R_MASK_PATH, 0) if R_MASK_PATH else None

        USE_FORWARD_ENERGY = not args["backward_energy"]

        # downsize image for faster processing
        h, w = img.shape[:2]
        if FLAGS.SHOULD_DOWNSIZE and w > FLAGS.DOWNSIZE_WIDTH:
            start_resize_time = time.time()
            img = resize(img, width=FLAGS.DOWNSIZE_WIDTH)
            if mask is not None:
                mask = resize(mask, width=FLAGS.DOWNSIZE_WIDTH)
            if rmask is not None:
                rmask = resize(rmask, width=FLAGS.DOWNSIZE_WIDTH)
            end_resize_time = time.time()
            resize_spend = timing(end_resize_time - start_resize_time)
            save_txt(outfddir+'/time.txt', resize_spend, string="\nresize: ")

        if args["resize"] or "resize" in FLAGS.ACTION:
            start_seam_carve_time = time.time()
            dy, dx = args["dy"], args["dx"]
            assert dy is not None and dx is not None
            output = seam_carve(img, dy, dx, mask, args["vis"])
            cv2.imwrite(IMG_OUT_PATH, output)
            end_seam_carve_time = time.time()
            seam_carve_spend = timing(end_seam_carve_time-start_seam_carve_time)
            save_txt(outfddir+'/time.txt', seam_carve_spend, string="\nseam_carve: ")

        # object removal mode
        elif args["remove"] or "remove" in FLAGS.ACTION:
            start_object_removal_time = time.time()
            assert rmask is not None
            output = object_removal(img, rmask, mask, args["vis"], args["hremove"])
            cv2.imwrite(IMG_OUT_PATH, output)
            end_object_removal_time = time.time()
            object_removal_spend = timing(end_object_removal_time-start_object_removal_time)
            save_txt(outfddir+'/time.txt', object_removal_spend, string="\nobject_removal: ")

    elif filterUseFolders(IMG_PATH, folders=FLAGS.image_format):   
        createFolder([outfddir])
        os.chdir(outfddir) 
        imgfdpath = os.getcwd()

        if FLAGS.save2Tiff_flg:
            start_save2Tiff_time = time.time()
            threads = []; usefile = []
            usefile = filterUseFolders(IMG_PATH, folders=FLAGS.image_format)
            array = belist(threads=FLAGS.threads, folders=usefile)
            for g in range(len(array)):
                for s in range(len(array[g])):
                    t = threading.Thread(target=save2Tiff, args=(IMG_PATH, [array[g][s]]))
                    t.start()
                    threads.append(t) 
                for thread in threads:
                    thread.join()
            end_save2Tiff = time.time()
            save2Tiff_spend = timing(end_save2Tiff - start_save2Tiff_time)
            save_txt(imgfdpath+'/time.txt', save2Tiff_spend, string="\nsave2Tiff: ")
            if FLAGS.backupall_flg or FLAGS.save2Tiff_bp_flg:
                backup_start_time = time.time()
                backup(path=imgfdpath, bkpath=FLAGS.backupfilepath, folders=[FLAGS.infolder])
                backup_end_time = time.time()
                backup_spend = timing(backup_end_time - backup_start_time)
                save_txt(FLAGS.backupfilepath+'/time.txt', backup_spend, string="\nsave2Tiff: ")
                logger.info("save2Tiff backup finish.")


        if FLAGS.genMask_flg:
            start_genMask_time = time.time()
            threads = []; usdfd = []
            usdfd = filterUseFolders(imgfdpath, folders=[FLAGS.infolder])
            array = belist(threads=FLAGS.threads, folders=usdfd, refers=FLAGS.default_size)
            for g in range(len(array)):
                for s in range(len(array[g])):
                    t = threading.Thread(target=genMask, args=(imgfdpath, FLAGS.target_size, [array[g][s][0]], [array[g][s][1]]))
                    t.start()
                    threads.append(t) 
                for thread in threads:
                    thread.join()
            end_genMask_time = time.time()
            genMask_spend = timing(end_genMask_time - start_genMask_time)
            save_txt(imgfdpath+'/time.txt', genMask_spend, string="\ngenMask: ")
            if FLAGS.backupall_flg or FLAGS.genMask_bp_flg:
                backup_start_time = time.time()
                backup(path=imgfdpath, bkpath=FLAGS.backupfilepath, folders=["mask"])
                backup_end_time = time.time()
                backup_spend = timing(backup_end_time - backup_start_time)
                save_txt(FLAGS.backupfilepath+'/time.txt', backup_spend, string="\ngenMask: ")
                logger.info("genMask backup finish.")


        if FLAGS.cropImg_flg:
            start_cropImg_time = time.time()
            threads = []; usdfd = []
            usdfd = filterUseFolders(imgfdpath, folders=[FLAGS.infolder])
            [usdfd.remove(f) for d in FLAGS.default_size for f in usdfd if str(d) in f]
            array = belist(threads=FLAGS.threads, folders=usdfd, refers=FLAGS.default_size)
            for g in range(len(array)):
                for s in range(len(array[g])):
                    t = threading.Thread(target=cropImg, args=(imgfdpath, [array[g][s][0]], [array[g][s][1]]))
                    t.start()
                    threads.append(t) 
                for thread in threads:
                    thread.join()
            end_cropImg_time = time.time()
            cropImg_spend = timing(end_cropImg_time - start_cropImg_time)
            save_txt(imgfdpath+'/time.txt', cropImg_spend, string="\ncropImg: ")
            if FLAGS.backupall_flg or FLAGS.cropImg_bp_flg:
                backup_start_time = time.time()
                backup(path=imgfdpath, bkpath=FLAGS.backupfilepath, folders=["crop"])
                backup_end_time = time.time()
                backup_spend = timing(backup_end_time - backup_start_time)
                save_txt(FLAGS.backupfilepath+'/time.txt', backup_spend, string="\ncropImg: ")
                logger.info("cropImg backup finish.")


        if FLAGS.carveImg_flg:
            start_carveImg_time = time.time()
            carveImg(imgfdpath, size=FLAGS.target_size, carves=FLAGS.default_size, framesize=FLAGS.frame_size)
            end_carveImg_time = time.time()
            carveImg_spend = timing(end_carveImg_time - start_carveImg_time)
            save_txt(imgfdpath+'/time.txt', carveImg_spend, string="\ncarveImg: ")
            if FLAGS.backupall_flg or FLAGS.carveImg_bp_flg:
                backup_start_time = time.time()
                backup(path=imgfdpath, bkpath=FLAGS.backupfilepath, folders=["seam"])
                backup_end_time = time.time()
                backup_spend = timing(backup_end_time - backup_start_time)
                save_txt(FLAGS.backupfilepath+'/time.txt', backup_spend, string="\ncarveImg: ")
                logger.info("carveImg backup finish.")
        
        if FLAGS.train_flg:
            start_trainData_time = time.time()
            trainData(imgfdpath, FLAGS.trainfilepath, size=FLAGS.train_size, frame=FLAGS.train_frame, num=FLAGS.trainNum)
            end_trainData_time = time.time()
            trainData_spend = timing(end_trainData_time - start_trainData_time)
            save_txt(imgfdpath+'/time.txt', trainData_spend, string="\ntrainData: ")
        
    else:
        logger.info("No image in path:" + IMG_PATH)


    end_time = time.time() 
    total_spend = timing(end_time - start_time)
    save_txt(outfddir+'/time.txt', total_spend, string="\ntotal: ")   
