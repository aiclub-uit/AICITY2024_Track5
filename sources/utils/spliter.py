import os
import random
import glob
import shutil
import argparse


def process(imglist, in_imgpath, in_labelpath, out_imgpath, out_labelpath):
    for fname in imglist:
        fname_woext = ".".join(fname.split(".")[:-1])

        img_pattern = "%s/%s.*" % (in_imgpath, fname_woext)
        fimgpath = glob.glob(img_pattern)[0]

        label_pattern = "%s/%s.*" % (in_labelpath, fname_woext)
        if len(glob.glob(label_pattern)) == 0:
            continue
        flabelpath = glob.glob(label_pattern)[0]
        flabelname = flabelpath.split("/")[-1]

        new_fimgpath = os.path.join(out_imgpath, fname)
        new_flabelpath = os.path.join(out_labelpath, flabelname)
        print("Copying %s to %s" % (fimgpath, new_fimgpath))
        shutil.copy(fimgpath, new_fimgpath)
        print("Copying %s to %s" % (flabelpath, new_flabelpath))
        shutil.copy(flabelpath, new_flabelpath)


def create_split_list(imgpath, train_percent, test_percent, val_percent):
    imglist = os.listdir(imgpath)
    imgnum = len(imglist)

    train_imgnum = int(imgnum * train_percent / 100)
    test_imgnum = int(imgnum * test_percent / 100)
    val_imgnum = int(imgnum * val_percent / 100)

    test_imglist = random.sample(imglist, test_imgnum)
    val_imglist = random.sample(imglist, val_imgnum)
    train_imglist = []
    for imgname in imglist:
        if imgname not in test_imglist:
            train_imglist.append(imgname)

    return train_imglist, val_imglist, test_imglist


def split(basepath, imgpath, labelpath, train_percent, test_percent, val_percent):
    if train_percent + test_percent + val_percent != 100:
        assert "percent sum not == 100"

    train_imgpath = os.path.join(basepath,"train_img")
    train_labelpath = os.path.join(basepath,"train_label")
    if not os.path.exists(train_imgpath):
        os.mkdir(train_imgpath)
    if not os.path.exists(train_labelpath):
        os.mkdir(train_labelpath)

    test_imgpath = os.path.join(basepath,"test_img")
    test_labelpath = os.path.join(basepath,"test_label")
    if not os.path.exists(test_imgpath):
        os.mkdir(test_imgpath)
    if not os.path.exists(test_labelpath):
        os.mkdir(test_labelpath)

    val_imgpath = os.path.join(basepath,"val_img")
    val_labelpath = os.path.join(basepath,"val_label")
    if not os.path.exists(val_imgpath):
        os.mkdir(val_imgpath)
    if not os.path.exists(val_labelpath):
        os.mkdir(val_labelpath)

    train_imglist, val_imglist, test_imglist = create_split_list(imgpath, train_percent, test_percent, val_percent)

    print("train_imglist num: ", len(train_imglist))
    print("test_imglist num: ", len(test_imglist))
    print("val_imglist num: ", len(val_imglist))

    process(train_imglist, imgpath, labelpath, train_imgpath, train_labelpath)
    process(test_imglist, imgpath, labelpath, test_imgpath, test_labelpath)
    process(val_imglist, imgpath, labelpath, val_imgpath, val_labelpath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split images and labels into train, test, val')
    parser.add_argument('--basepath', type=str, default="data/aicity2024_track5_train/", help='Base path of the dataset')
    parser.add_argument('--imgpath', type=str, default="images", help='Path to the image folder')
    parser.add_argument('--labelpath', type=str, default="simpjson", help='Path to the label folder')
    parser.add_argument('--train_percent', type=int, default=80, help='Percentage of train set')
    parser.add_argument('--test_percent', type=int, default=20, help='Percentage of test set')
    parser.add_argument('--val_percent', type=int, default=0, help='Percentage of val set')
    args = parser.parse_args()

    basepath = args.basepath
    imgpath = os.path.join(basepath,args.imgpath)
    labelpath = os.path.join(basepath,args.labelpath)
    train_percent = args.train_percent
    test_percent = args.test_percent
    val_percent = args.val_percent
    split(basepath, imgpath, labelpath, train_percent, test_percent, val_percent)
