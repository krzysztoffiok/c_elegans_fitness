This repository contains a copy of MASK-RCNN (https://github.com/matterport/Mask_RCNN) model modified for the purpose of carrying out C. Elegans fitness analysis described in paper (in preparation):

*"Fitness analysis of Caenorhabditis elegans using Convolutional Neural Network"*

Authors: Joanna K. Palka1, Krzysztof Fiok2, Weronika Antoł1, Zofia M. Prokop1

1 Jagiellonian University in Krakow, Institute of Environmental Sciences\
2 University of Central Florida, Industrial Engineering & Management Systems


The whole repository is published under MIT License (please refer to the [License file](https://github.com/krzysztoffiok/c_elegans_fitness/blob/master/LICENSE)).

In due course full description of usage will appear here.

Trained model weights are available in the release section (0.1 release).

The code is written in Python3 and requires GPU computing machine for achieving reasonable performance.

Installation:
model-specific-problem-causing package is called PyCoCoTools. It is provided in a proper version in this repo as "modified_pycoco_files.zip". Before using the model, please unzip this file so that the "coco" folder is on the same level as your .py files. Next run:\
cd coco/PythonAPI<br/>
python setup.py build_ext install<br/>

Other dependecies are described in (https://github.com/matterport/Mask_RCNN), and are met e.g. by [Google Colaboratory](https://colab.research.google.com).<br/>

Example results of the model are presented below:<br/>

 

Example use:\
To analyze images you can run:\
python3 ce_bbx.py inference --image_folder=/path/to/images/for/inference --DMC=0.9 --NMS=0.6 --model=/path/to/model_weights<br/>

To compute modified MS COCO metrics:\
python3 ce_bbx.py evaluate --dataset=/path/to/evaluation_dataset --DMC=0.9 --NMS=0.6 --model=/path/to/model_weights<br/>

To draw Precision Recall Curve (PRC):\
python3 ce_bbx.py evaluate_PRC ----dataset=/path/to/evaluation_dataset  --model=/path/to/model_weights<br/>

To train your own model:\
python3 ce_bbx.py train --dataset=/path/to/train_dataset --model=path_to_initial_model_weights<br/>

Training the model on segmantation dataset is possible:\
python3 ce_segmentation.py train --dataset=/path/to/dataset --model=/path/to/model_weights<br/>


