import os
import cv2
import argparse
from tqdm import tqdm

def extract_frame(inpath, outpath, video_name):
    cap = cv2.VideoCapture(inpath)
    if (cap.isOpened()== False): 
        print("Error opening video file") 
    
    frame_id = 1
    while(cap.isOpened()): 
        ret, frame = cap.read() 
        if ret == True: 
            fname = os.path.join(outpath, "%s_%s.jpg" %(video_name, frame_id))
            cv2.imwrite(fname, frame)
        else: 
            break
        frame_id += 1

def main(video_path: str, images_path: str):

    for video_name in tqdm(os.listdir(video_path)):
        inpath = os.path.join(video_path, video_name)
        outpath = images_path
        video_name = str(int(video_name.split(".mp4")[0]))+".mp4"
        extract_frame(inpath, outpath, video_name)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert videos to images')
    parser.add_argument('--basepath', type=str, default="data/aicity2024_track5_train/", help='Base path of the dataset')
    args = parser.parse_args()
    
    basepath = args.basepath
    video_path = os.path.join(basepath, "videos")
    images_path = os.path.join(basepath, "images")
    os.makedirs(images_path, exist_ok=True)

    main(video_path, images_path)