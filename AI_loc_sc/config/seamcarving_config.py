import os

class FLAGS(object):
    
    ### General parameter
    # Path to image
    infile = "" 
    infolder = "coco_val_2014"
    inpath = "/media/m0629010/24683ec1-80d0-4847-a85e-a57684d3c2a7/"
    infilepath = os.path.join(inpath, infolder, infile)
    # Output file name
    outfile = "" 
    outfolder = "coco_val_2014_sc"
    outpath = "/media/m0629010/24683ec1-80d0-4847-a85e-a57684d3c2a7/Images"
    outfilepath = os.path.join(outpath, outfolder, outfile)
    # Path to (protective) mask
    maskfile = "" 
    maskfolder = ""
    maskpath = ""
    maskfilepath = os.path.join(maskpath, maskfolder, maskfile)
    # Path to removal mask
    rmaskfile = "" 
    rmaskfolder = ""
    rmaskpath = ""
    rmaskfilepath = os.path.join(rmaskpath, rmaskfolder, rmaskfile)


    ### Image setting
    image_format = ["jpg", "tiff", "png"]

    ACTION = "resize"                         # resize or remove
    ### Single image parameter
    SHOULD_DOWNSIZE = True                    # if True, downsize image for faster carving
    DOWNSIZE_WIDTH = 500                      # resized image width if SHOULD_DOWNSIZE is True
    dy = -10
    dx = -10
    ### Folder images parameter
    default_size = [160, 288, 416, 544]       # target_size = [128, 256, 384, 512]
    target_size = 32
    frame_size = [8, 16, 24, 32]
    threads = 16
    ### Get Train folder parameter
    trainfolder = "images_tain"
    trainpath = "/media/steven/833779b7-e371-4057-b024-d5c66335cba9/Images"
    trainfilepath = os.path.join(trainpath, trainfolder)
    train_size = 160
    train_frame = 8
    trainNum = 2


    # func control
    save2Tiff_flg = True
    genMask_flg = True
    cropImg_flg = True
    carveImg_flg = True
    train_flg = False

    # backup control 
    backupfile = "coco_test_2014_bp"
    backuppath = "/media/steven/833779b7-e371-4057-b024-d5c66335cba9/"
    backupfilepath = os.path.join(backuppath, backupfile)    
    backupall_flg = False
    save2Tiff_bp_flg = False
    genMask_bp_flg = False
    cropImg_bp_flg = False
    carveImg_bp_flg = False