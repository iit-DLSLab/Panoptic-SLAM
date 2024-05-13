# Panoptic-SLAM: Visual SLAM in Dynamic Environments using Panoptic Segmentation

This work presents Panoptic-SLAM, a visual SLAM system robust to dynamic environments, even in the presence of unknown objects. It uses panoptic segmentation to filter dynamic objects from the scene during the state estimation process. Panoptic-SLAM is based on ORB-SLAM3, a state-of-the-art SLAM system for static environments.

Paper: **[PDF](https://arxiv.org/pdf/2405.02177)**.

Demonstration video: **[Video](https://www.youtube.com/watch?v=BNCKWLrMk8I)**

## License
Panoptic-SLAM is released under a GPLv3 License
if you use Panoptic-SLAM in an academic work, please cite:
```
@misc{abati2024panopticslam,
      title={Panoptic-SLAM: Visual SLAM in Dynamic Environments using Panoptic Segmentation}, 
      author={Gabriel Fischer Abati and João Carlos Virgolino Soares and Vivian Suzano Medeiros and Marco Antonio Meggiolaro and Claudio Semini},
      year={2024},
      eprint={2405.02177},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## Build Panoptic-SLAM on Docker
For easy installation of our system, we provide a dockerfile to install all dependencies from Panoptic-SLAM. You can install docker following the instruction from [here](https://docs.docker.com/engine/install/ubuntu/). </br>
First, clone the repository:
```
git clone https://github.com/iit-DLSLab/Panoptic-SLAM
```
To build the docker image, run the following command. Note the docker image will create a virtual environment with ubuntu 20.04, CUDA 11.6, cuddn8 and python 3.8. The building process might take some time.
```
sudo docker build -t panoptic_slam -f docker/dockerfile .
```

### Running Panoptic-SLAM on Docker
To run Panoptic-SLAM inside the docker, we provide a docker compose file for easy access to the docker container. If you need to install docker compose, there is a download bash file in ```docker/install_docker_compose.sh```. To map the dataset data in the host machine with the container, create a folder called Dataset and insert your data there. Then follow these commands
```
xhost +
docker compose -f docker/docker-compose.yaml run panoptic_slam bash
source .venv/bin/activate
./panoptic_slam Vocabulary/ORBvoc.txt.tar.gz /home/Dataset/<Path To Settings> /home/Dataset/<Path to Dataset> /home/Dataset/<Path To Association file>
```
The trajectory files can be retrived from the docker container by moving them to the  ```Output``` folder.


## Build Panoptic-SLAM on Host Machine
### 1. Prerequisistes
#### C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

#### Pangolin
We use Pangolin for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

#### OpenCV
We use OpenCV to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. Required at leat 3.0. Tested with OpenCV 3.2.0 and 4.4.0.

#### Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. Required at least 3.1.0.

#### DBoW2 and g2o (Included in Thirdparty folder)

We use modified versions of the DBoW2 library to perform place recognition and g2o library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the Thirdparty folder.

#### Python
Required for panoptic-Segmentation inference and alignment of the trajectory with the ground truth. The tested python version is 3.8.10.

#### detectron2
we use detectron2 library to perform panoptic-segmentation inference in python and use the C-python-API to communicate the inference module with the SLAM system. The detectron2 framework can be installed from: https://github.com/facebookresearch/detectron2. We are using the version 0.5 that can be found here: https://github.com/facebookresearch/detectron2@v0.5


### 2. Building Panoptic-SLAM library and examples
Clone the repository
```
git clone https://github.com/iit-DLSLab/Panoptic-SLAM
```

We provide a script build.sh to build the Thirdparty libraries and Panoptic-SLAM. The CMakeLists.txt is configured to run with CUDA 11.6 and python3.8
```
cd Panoptic-SLAM
chmod +x build.sh
./build.sh
```

### 3. TUM RGB-D Dataset Example
- Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.
- Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools):
```
python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
```
- Execute the following command. Change <Path To Settings> to the path for the TUM1.yaml, TUM2.yaml or TUM3.yaml files. Change <Path to Dataset> to the uncompressed sequence folder. Change <Path To Association file> to the path to the corresponding associations file.
```
./panoptic_slam Vocabulary/ORBvoc.txt <Path To Settings> <Path to Dataset> <Path To Association file>
```

## Acknowledgements
Our code builds on [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3).

## Maintainer

This repository is maintained by [Gabriel Fischer](https://github.com/git-gfischer) and [João Soares](https://github.com/virgolinosoares).
