python3 sources/utils/converter.py --basepath data/aicity2024_track5_train-real/ --gtpath gt.txt --outpath simpjson
python3 sources/utils/extracter.py --basepath data/aicity2024_track5_train-real/ 
python3 sources/utils/spliter.py --basepath data/aicity2024_track5_train-real/ 
python3 sources/utils/simpjson2coco.py --basepath data/aicity2024_track5_train-real/ 