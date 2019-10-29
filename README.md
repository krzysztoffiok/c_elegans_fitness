This repository contains a copy of [Mask R-CNN](https://github.com/matterport/Mask_RCNN) model modified for the purpose of carrying out C. Elegans fitness analysis described in paper (in preparation):

*"Fitness analysis of Caenorhabditis elegans using Convolutional Neural Network"*

Authors: Joanna K. Palka1, Krzysztof Fiok2, Weronika Antoł1, Zofia M. Prokop1

1 Jagiellonian University in Krakow, Institute of Environmental Sciences\
2 University of Central Florida, Industrial Engineering & Management Systems


The whole repository is published under MIT License (please refer to the [License file](https://github.com/krzysztoffiok/c_elegans_fitness/blob/master/LICENSE)).

In due course full description of usage will appear here.

Trained model weights are available in the [release section](https://github.com/krzysztoffiok/c_elegans_fitness/releases).

The code is written in Python3 and requires GPU computing machine for achieving reasonable performance.

## Installation:
Model-specific-problem-causing package is called PyCoCoTools. It is provided in a proper version in this repo as "modified_pycoco_files.zip". Before using the model, please unzip this file so that the "coco" folder is on the same level as your .py files. Next run:\
cd coco/PythonAPI<br/>
python setup.py build_ext install<br/>

The model is written in Keras and Tensorflow. Dependecies are described more precisely in [Mask R-CNN](https://github.com/matterport/Mask_RCNN), and are met e.g. by [Google Colaboratory](https://colab.research.google.com) most of the time.<br/>

Datasets with bounding box annotations and precise instance segmentation masks used for training and evaluation of the model are provided in .zip files in the "datasets" folder. It is advised to unpack them at the same level as .py files.<br/>

Description of model parameters along with adopted values is available in mrcnn/config.py.<br/>

Changes in visualisation of detections by the model can be done in mrcnn/visualize.py, especially in the "display_instances" function.<br/>

## Example results are presented below:<br/>

<img src="https://github.com/krzysztoffiok/c_elegans_fitness/blob/master/example_results/result_K03_2.jpg" width=640 height=512><br/>
<img src="https://github.com/krzysztoffiok/c_elegans_fitness/blob/master/example_results/result_K05_1.jpg" width=640 height=512><br/>  

## Example use:<br/>
To analyze images you can run:\
python3 ce_bbx.py inference --image_folder=/path/to/images/for/inference --DMC=0.9 --NMS=0.6 --model=/path/to/model_weights<br/>

The model returns .csv file with number of instances of glowing and not glowing c.elegans per image, another .csv file with locations of found c.elegans (rectangles) and a folder with images after inference.<br/>

To compute modified MS COCO metrics:\
python3 ce_bbx.py evaluate --dataset=/path/to/evaluation_dataset --DMC=0.9 --NMS=0.6 --model=/path/to/model_weights<br/>

To draw Precision Recall Curve (PRC):\
python3 ce_bbx.py evaluate_PRC --dataset=/path/to/evaluation_dataset  --model=/path/to/model_weights<br/>

To train your own model:\
python3 ce_bbx.py train --dataset=/path/to/train_dataset --model=path_to_initial_model_weights<br/>

Training the model on segmantation dataset is possible:\
python3 ce_segmentation.py train --dataset=/path/to/dataset --model=/path/to/model_weights<br/>

If you wish to train your model from MS CoCo weights, please download them from the original repository  [Mask R-CNN](https://github.com/matterport/Mask_RCNN).

## Citation:<br/>
If you decide to use here published version of the Mask R-CNN model, model weights or labelled or unlabelled images from our datasets please cite our work in the following manner:
(please contact us directly at this time since the paper is still in preparation).

