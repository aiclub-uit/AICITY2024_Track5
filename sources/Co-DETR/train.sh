export PYTHONPATH=${PYTHONPATH}:/base
python3 tools/train.py /base/helios-cfg/co_dino_5scale_swin_large_16e_o365tococo.py --work-dir helios
