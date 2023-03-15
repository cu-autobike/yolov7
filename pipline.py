import csv
import os
import json
import shutil
import cv2
import random

# Get the list of classes for LISA
def getLabelsLISA():
    direct = "../Downloads/LISA/Annotations/Annotations/"
    sub_dir = os.listdir(direct)

    LISA_classes = set()

    for sdir in sub_dir:
        if sdir=='dayTrain' or sdir == "nightTrain":
            sub_sub_dir = os.listdir(direct+sdir)
            for ssdir in sub_sub_dir:
                for annotfile in os.listdir(direct+"/"+sdir+"/"+ssdir):
                    annot = open(direct+"/"+sdir+"/"+ssdir+"/"+annotfile)
                    annot_csv = csv.reader(annot, delimiter=';')
                    next(annot_csv)
                    for row in annot_csv:
                        LISA_classes.add(row[1])
        else:
            for annotfile in os.listdir(direct+"/"+sdir):
                annot = open(direct+"/"+sdir+"/"+annotfile)
                annot_csv = csv.reader(annot, delimiter=';')
                next(annot_csv)
                for row in annot_csv:
                    LISA_classes.add(row[1])

    
    print(LISA_classes)

# filter out classes out of coco we will not use
def filterCOCO():

    coco_to_raph = {
        '0' : 0,
        '1' : 1,
        '2' : 2,
        '3' : 3,
        '4': -1,
        '5' : 4,
        '6': -1,
        '7' : 5,
        '8': -1,
        '9' : 6,
        '10' : 7,
        '11' : 8,
        '12' : 9,
        '13' : 10,
        '14': -1,
        '15' : 11,
        '16' : 12,
        '17' : 13,
        '18': -1,
        '19': -1,
        '20': -1,
        '21': -1,
        '22': -1,
        '23': -1,
        '24': -1,
        '25': -1,
        '26': -1,
        '27': -1,
        '28': -1,
        '29': -1,
        '30': -1,
        '31': -1,
        '32': -1,
        '33': -1,
        '34': -1,
        '35': -1,
        '36': -1,
        '37': -1,
        '38': -1,
        '39': -1,
        '40': -1,
        '41': -1,
        '42': -1,
        '43': -1,
        '44': -1,
        '45': -1,
        '46': -1,
        '47': -1,
        '48': -1,
        '49': -1,
        '50': -1,
        '51': -1,
        '52': -1,
        '53': -1,
        '54': -1,
        '55': -1,
        '56': -1,
        '57': -1,
        '58': -1,
        '59': -1,
        '60': -1,
        '61': -1,
        '62': -1,
        '63': -1,
        '64': -1,
        '65': -1,
        '66': -1,
        '67': -1,
        '68': -1,
        '69': -1,
        '70': -1,
        '71': -1,
        '72': -1,
        '73': -1,
        '74': -1,
        '75': -1,
        '76': -1,
        '77': -1,
        '78': -1,
        '79': -1
    }

    for p in ["train2017", "val2017", "test2017"]:
        
        path = coco_path + "/" + p
        keepers = dict()

        all_files = os.listdir(path)

        c = 0

        for f in all_files:
            to_keep = []
            annot_file = open(path+"/"+f)
            annot_data = csv.reader(annot_file, delimiter = ' ')
            for row in annot_data:
                if coco_to_raph[row[0]] != -1:
                    row[0] = str(coco_to_raph[row[0]])
                    to_keep.append(" ".join(row))
            
            if to_keep!=[]:
                keepers[f] = "\n".join(to_keep)

        #dump content of keepers in dict
        with open(raph_path+"/"+p+".json", "w") as f:
            json.dump(keepers, f)

# Read json dumped file and copying images to the right location in raph+
def moveImages():

    raph_path = "/home/autobike/yolov7/raph/labels/"

    for p in ["train2017", "val2017"]:
        
        path = raph_path + p+".json"

        json_file = open(path)

        data = json.load(json_file)

        l = list(data.keys())

        images_path = "/home/autobike/yolov7/raph/images/"

        for filename in l:
            imagename = filename.replace(".txt", ".jpg")
            shutil.copy("/home/autobike/yolov7/coco/images/" + p + "/" + imagename , images_path + p)
            shutil.copy("/home/autobike/yolov7/coco/labels/" + p + "/" + filename , raph_path + p)

# The full coco pipeline
def COCOPipeline():
    filterCOCO()
    moveImages()

# Outputs a jason with all the LISA annotation converted to YOLOv7 annotation files
def convertingLISA():

    LISA_to_Raph = {
        'warning':14,
        'stopLeft':15,
        'warningLeft':16,
        'go':17,
        'goLeft':18,
        'stop':19,
        'goForward':20
    }

    direct = "../Downloads/LISA/Annotations/Annotations/"
    sub_dir = os.listdir(direct)

    # key is path to image file, value is all <class, x_center, y_center, width, height> for that image file
    final_dict = dict()
    c = 0

    for sdir in sub_dir:
        if sdir=='dayTrain' or sdir == "nightTrain":
            sub_sub_dir = os.listdir(direct+sdir)
            for ssdir in sub_sub_dir:
                for annotfile in os.listdir(direct+"/"+sdir+"/"+ssdir):
                    annot = open(direct+"/"+sdir+"/"+ssdir+"/"+annotfile)
                    annot_csv = csv.reader(annot, delimiter=';')
                    next(annot_csv)
                    for row in annot_csv:
                        c+=1
                        # path to the corresponding image
                        abs_path = "/home/autobike/Downloads/LISA/"+sdir+"/"+sdir+"/"+ssdir+"/frames/"+row[0].split('/')[1]
                            
                        # opening img to normalize the coordinates
                        img_shape = cv2.imread(abs_path).shape

                        # changing the class from one of 7 LISA class to the corresponding class in raph.yaml
                        raph_class = LISA_to_Raph[row[1]]

                        # Changing the coordinates from Upper left corner X', 'Upper left corner Y', 'Lower right corner X', 'Lower right corner Y' to <x_center, y_center, width, height>
                        up_left_x, up_left_y, low_right_x, low_right_y = float(row[2]), float(row[3]), float(row[4]), float(row[5])
                        width = abs(up_left_x-low_right_x)
                        height = abs(up_left_y-low_right_y)
                        center_x = up_left_x+width/2
                        center_y = up_left_y+height/2
                        data_to_save = (raph_class, center_x/img_shape[1], center_y/img_shape[0], width/img_shape[1], height/img_shape[0])

                        # saving new data into a json
                        if abs_path in final_dict:
                            final_dict[abs_path].append(data_to_save)
                        else:
                            final_dict[abs_path] = [data_to_save]

        else:

            for annotfile in os.listdir(direct+"/"+sdir):
                annot = open(direct+"/"+sdir+"/"+annotfile)
                annot_csv = csv.reader(annot, delimiter=';')
                next(annot_csv)
                for row in annot_csv:
                    c+=1

                    LISA_dir = row[0].split('/')[1].split("--")[0]
                    abs_path = "/home/autobike/Downloads/LISA/"+LISA_dir+"/"+LISA_dir+"/frames/"+row[0].split('/')[1]

                    # opening img to normalize the coordinates
                    img_shape = cv2.imread(abs_path).shape

                    # changing the class from one of 7 LISA class to the corresponding class in raph.yaml
                    raph_class = LISA_to_Raph[row[1]]

                    # Changing the coordinates from Upper left corner X', 'Upper left corner Y', 'Lower right corner X', 'Lower right corner Y' to <x_center, y_center, width, height>
                    up_left_x, up_left_y, low_right_x, low_right_y = float(row[2]), float(row[3]), float(row[4]), float(row[5])
                    width = abs(up_left_x-low_right_x)
                    height = abs(up_left_y-low_right_y)
                    center_x = up_left_x+width/2
                    center_y = up_left_y+height/2
                    data_to_save = (raph_class, center_x/img_shape[1], center_y/img_shape[0], width/img_shape[1], height/img_shape[0])

                    # saving new data into a json
                    if abs_path in final_dict:
                        final_dict[abs_path].append(data_to_save)
                    else:
                        final_dict[abs_path] = [data_to_save]
            
            print(f"{100*c/226353}%")

    f = open("LISA_to_raph.json", "w")
    data = json.dump(final_dict, f)

# We have json with images and annotations, we now need to move them to raph folder
def moveLISAAnnotations():
    f = open("LISA_to_raph.json")
    data = json.load(f)

    for image in data.keys():

        if random.random()<0.85:
            # add image to train
            shutil.copy(image, "/home/autobike/yolov7/raph/images/train2017")

            annot = []

            for single_annot in data[image]:
                temp = ""
                for n in single_annot:
                    temp+=str(n)+" "
                
                annot.append(temp[:-1])

            with open("/home/autobike/yolov7/raph/labels/train2017/"+image.split("/")[-1][:-3]+"txt", "w") as f:
                
                for a in annot:
                    f.write(a)

        elif random.random() < 0.32:
            # add to val

            shutil.copy(image, "/home/autobike/yolov7/raph/images/val2017")

            annot = []

            for single_annot in data[image]:
                temp = ""
                for n in single_annot:
                    temp+=str(n)+" "
                
                annot.append(temp[:-1])

            with open("/home/autobike/yolov7/raph/labels/val2017/"+image.split("/")[-1][:-3]+"txt", "w") as f:
                
                for a in annot:
                    f.write(a)
        else:
            # add to test
            shutil.copy(image, "/home/autobike/yolov7/raph/images/test2017")

            annot = []

            for single_annot in data[image]:
                temp = ""
                for n in single_annot:
                    temp+=str(n)+" "
                
                annot.append(temp[:-1])

            with open("/home/autobike/yolov7/raph/labels/test2017/"+image.split("/")[-1][:-3]+"txt", "w") as f:
                
                for a in annot:
                    f.write(a)

def make_txt_with_all_images_and_annotations_this_means_making_val2017_and_train2017_and_testdev2017():
    f = open("/home/autobike/yolov7/raph/test-dev2017.txt", "w")
    f.close()

    with open("/home/autobike/yolov7/raph/train2017.txt", "w") as f:
        pass

    with open("/home/autobike/yolov7/raph/val2017.txt", "w") as f:
        pass

make_txt_with_all_images_and_annotations_this_means_making_val2017_and_train2017_and_testdev2017() 

def pipeline():
    # making raph directory where we store the data
    os.mkdir("raph")
    os.mkdir("raph/annotations")
    os.mkdir("raph/images")
    os.mkdir("raph/images/test2017")
    os.mkdir("raph/images/train2017")
    os.mkdir("raph/labels")
    os.mkdir("raph/labels/train2017")
    os.mkdir("raph/labels/val2017")

    COCOPipeline()
