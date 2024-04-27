FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
ENV CUDA_HOME=/usr/local/cuda-11.3/
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
    TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    FORCE_CUDA="1"

WORKDIR /base

RUN ln -sf /usr/share/zoneinfo/Asia/Ho_Chi_Minh /etc/localtime

RUN apt-get update && apt-get install -y python3 python3-pip cmake wget llvm
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx

RUN pip3 install torch==1.11.0
RUN pip3 install torchvision==0.12.0
#RUN pip3 install mmengine==0.10.3
RUN pip3 install terminaltables==3.1.10
RUN pip3 install pycocotools==2.0.7
RUN pip3 install fairscale==0.4.13
RUN pip3 install timm==0.9.16
RUN pip3 install yapf==0.40.2
#RUN pip3 install mmcv==1.3.17
#RUN pip3 install mmcv-full==1.3.17
RUN pip3 install -U openmim
RUN apt install -y wget
RUN wget https://bootstrap.pypa.io/get-pip.py -O ./get-pip.py
RUN python3 ./get-pip.py
RUN pip3 install yapf==0.40.1

RUN pip3 install --no-cache-dir --upgrade pip wheel setuptools
RUN mim install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
RUN pip3 install scipy==1.10.1
#RUN mim install mmcv==1.3.17
#RUN pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu112/torch1.13.1/index.html
#RUN mim install mmcv-full==1.3.17

#RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py39_24.1.2-0-Linux-x86_64.sh -O /tmp/miniconda.sh
#RUN /bin/bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh
#RUN /opt/conda/bin/conda config --set ssl_verify False
#RUN /opt/conda/bin/conda create -n env1 python=3.8
#RUN echo '/opt/conda/bin/activate env1' > ~/.bashrc
#ENV PATH="/opt/conda/bin:${PATH}"
#RUN conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
#RUN conda install pip
#RUN /opt/conda/bin/activate env1 && pip install --no-cache-dir mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
RUN apt-get install -y python3-magic
RUN pip3 install python-magic
RUN pip install ensemble-boxes==1.0.9
# CMD tail -f /dev/null
