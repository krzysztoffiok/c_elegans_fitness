"""
Mask R-CNN Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Modified for CElegans fitness assessment by Krzysztof Fiok
------------------------------------------------------------

Example use:

To analyze images:
python3 ce_bbx.py inference --image_folder=/path/to/images/for/inference --DMC=0.9 --NMS=0.6 --model=/path/to/model_weights

To compute modified MS COCO metrics:
python3 ce_bbx.py evaluate --dataset=/path/to/evaluation_dataset --DMC=0.9 --NMS=0.6 --model=/path/to/model_weights

To draw Precision Recall Curve (PRC):
python3 ce_bbx.py evaluate_PRC ----dataset=/path/to/evaluation_dataset  --model=/path/to/model_weights

To train your own model:
python3 ce_bbx.py train --dataset=/path/to/train_dataset --model=path_to_initial_model_weights

"""

import os
import sys
import time
import numpy as np
import imgaug
import skimage.draw
from mrcnn import visualize
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "", "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = ""


############################################################
#  Configurations
############################################################


class CElegansConfig(Config):
    """Configuration for training on .
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "CE"

    # Depends on GPU RAM
    IMAGES_PER_GPU = 1

    # (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # CE has 2 classes

    RPN_NMS_THRESHOLD = 0.9

    DETECTION_MIN_CONFIDENCE = 0.7


############################################################
#  Dataset
############################################################

class CElegansDataset(utils.Dataset):
    def load_celegans(self, dataset_dir, subset, class_ids=None, return_coco=False):
        """Load a subset of the CE dataset.
        dataset_dir: The root directory of the CE dataset.
        subset: What to load (train, val)
        class_ids: If provided, only loads images that have the given classes.
        return_coco: If True, returns the COCO object.
        """

        coco = COCO("{}/annotations/{}.json".format(dataset_dir, subset))
        image_dir = "{}/{}".format(dataset_dir, subset)

        class_ids = sorted(coco.getCatIds())
        image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_mask(self, image_id):
        """Load instance masks for the given image.
        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "coco":
            return super(CElegansDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "coco.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info["height"],
                                   image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(CElegansDataset, self).load_mask(image_id)

    # The following two functions are from pycocotools with a few changes.
    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m


############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco_PRC(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)
        r["class_ids"] = [x-1 for x in r["class_ids"]]

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate

    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids

    cocoEval.params.iouThrs = [0.1]
    cocoEval.params.areaRng = [[0, 10000000000.0]]
    cocoEval.params.maxDets = [100]

    cocoEval.evaluate()
    cocoEval.accumulate()

    precision = cocoEval.eval['precision'][0, :, 0, 0, 0]
    recall = cocoEval.params.recThrs
    plt.plot(recall, precision, 'ro')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('PRC IoU 0,5')
    plt.savefig(fname='PRC' + str(limit) + '.jpg')

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)


    # Pick COCO images from the dataset
    # image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)
        # r["class_ids"] = [x-1 for x in r["class_ids"]]

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)
    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()

    print('Original COCO metrics')
    sumcoco = cocoEval.summarize_coco()
    sumcoco = pd.DataFrame(sumcoco)

    print('Original PASCAL VOC metrics')

    sumvoc = cocoEval.summarize_voc()
    sumvoc = pd.DataFrame(sumvoc)

    sumcoco.to_csv('output_coco_%s.csv' % args.model[-6:])
    sumvoc.to_csv('output_voc_%s.csv' % args.model[-6:])


############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    parser.add_argument('--image_folder', required=False,
                        metavar="folder path",
                        help='Folder path with images for inference')
    parser.add_argument('--results', required=False, default='', help="String added at end of results path")
    parser.add_argument('--DMC', required=False, default=0.95, type=float, help="Provide Detection Max Confidence")
    parser.add_argument('--NMS', required=False, default=0.5, type=float, help="Provide Non Maximum Suppression")
    args = parser.parse_args()

    dmc = args.DMC
    nms = args.NMS

    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CElegansConfig()
    else:
        import ce_segmentation
        config = ce_segmentation.CElegansInferenceConfig()
        config.DETECTION_MIN_CONFIDENCE = float(dmc)
        config.DETECTION_NMS_THRESHOLD = float(nms)

    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()
    else:
        model_path = args.model

    # Load weights
    if args.model.lower() == "coco":
        model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

    else:
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset
        dataset_train = CElegansDataset()
        dataset_train.load_celegans(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = CElegansDataset()
        dataset_val.load_celegans(args.dataset, "val")
        dataset_val.prepare()

        # Image Augmentation
        augmentation = imgaug.augmenters.CoarseDropout(p=0.1, size_percent=0.3)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=50,
                    layers='heads',
                    augmentation=None)

        print("Training 4+ network layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=100,
                    layers='4+',
                    augmentation=None)

        print("Training all network layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 5,
                    epochs=150,
                    layers='all',
                    augmentation=None)

    elif args.command == "inference":

        class_names = ['BG', 'CE', 'CE_glow', 'dec', 'dead', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light',
                       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                       'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                       'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                       'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                       'kite', 'baseball bat', 'baseball glove', 'skateboard',
                       'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                       'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                       'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                       'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                       'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                       'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                       'teddy bear', 'hair drier', 'toothbrush']

        column_names = ['CE', 'CE_glow', 'file_name']
        bbx_info_list = pd.DataFrame(columns=['y1', 'x1', 'y2', 'x2', 'file_name'])
        image_file_list = []
        output_path = (os.path.join(ROOT_DIR, 'result_%s' % str(args.model), args.results))
        try:
            os.makedirs(output_path)
        except FileExistsError:
            print('----- The output folder already exists, overwriting ----')

        CE_list = []
        CE_glow_list = []

        for root, dirs, files in os.walk(args.image_folder, topdown=False):
            for name in files:
                print(name)
                input_file_path = os.path.join(root, name)
                image = skimage.io.imread(input_file_path)
                # Run detection
                results = model.detect([image], verbose=1)
                # Visualize results
                r = results[0]

                class_count, bbx_info = visualize.display_instances(output_path, name, image, r['rois'],
                                                                    r['masks'], r['class_ids'],
                                                                    class_names, r['scores'], show_mask=True)
                image_file_list.append(str(name))
                class_count = class_count.T
                print(class_count)
                CE_list.append((class_count.iloc[0, 0]))
                CE_glow_list.append(str(class_count.iloc[0, 1]))

                bbx_info_list = pd.concat([bbx_info_list, bbx_info], axis=0, sort=True)

        class_count_list = {"file_name": image_file_list, "CE": CE_list, "CE_glow": CE_glow_list}
        class_count_list = pd.DataFrame(class_count_list)

        class_count_list.to_csv(os.path.join(output_path, 'class_count.csv'))
        bbx_info_list = bbx_info_list.sort_values(by=['file_name'])
        bbx_info_list.to_csv(os.path.join(output_path, 'bbx_info.csv'))
        print('The results were saved to: ', output_path)

    elif args.command == "eval_table":
        dataset_val = CElegansDataset()
        coco = dataset_val.load_celegans(args.dataset, "val")
        dataset_val.prepare()
        evaluate(model, dataset_val, coco, "bbox", limit=int(args.limit))

    elif args.command == "evaluate":
        dataset_val = CElegansDataset()
        coco = dataset_val.load_celegans(args.dataset, "val", return_coco=True)
        dataset_val.prepare()
        evaluate_coco_PRC(model, dataset_val, coco, "bbox", limit=int(args.limit))

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
