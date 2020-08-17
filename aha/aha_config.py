"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("Mask_RCNN")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, mrcnn_utils

# Path to trained weights file
COCO_WEIGHTS_PATH = "weights\mask_rcnn_coco.h5"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "logs"

############################################################
#  Configurations
############################################################


class AHAConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "aha"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 15  # Background + classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.1

############################################################
#  Dataset
############################################################

class AHADataset(mrcnn_utils.Dataset):

    def load_aha(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("Part", 1, "Beam")
        self.add_class("Part", 2, "Rod")
        self.add_class("Part", 3, "Cable")
        self.add_class("Part", 4, "Strut")
        self.add_class("Part", 5, "NodeDock")
        self.add_class("Part", 6, "Joint")
        self.add_class("Part", 7, "JointSupport")
        self.add_class("Part", 8, "RigidSupport")
        self.add_class("Part", 9, "RollerSupport")
        self.add_class("Part", 10, "Pulley")
        self.add_class("Part", 11, "RigidJointTwo")
        self.add_class("Part", 12, "RigidJointThree")
        self.add_class("Part", 13, "RigidJointFour")
        self.add_class("Part", 14, "Rope")
        self.add_class("Part", 15, "BeamExtender")


        # Train or validation dataset?
        assert subset in ["train", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "aha_labels.json")))
        annotations = list(annotations.values())  # don't need the dict keys


        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            # if type(a['regions']) is dict:
            #     polygons = [r['shape_attributes'] for r in a['regions'].values()]
            # else:

            # Reference : https://github.com/matterport/Mask_RCNN/issues/372
            polygons = []
            for r in a[r'regions']:
                if a[r'regions']:
                    if r:
                    # if r['shape_attributes']:
                    # print(r['shape_attributes'])
                        polygons.append(r['shape_attributes'])

            # polygons = [r['shape_attributes'] for r in a[r'regions'] if a[r'regions']]
            parts = []
            for s in a[r'regions']:
                if a[r'regions']:
                    if s:
                        parts.append(s[r'region_attributes'])

            # parts = [s['region_attributes'] for s in a[r'regions'] if a[r'regions']]
            # print(parts)
            num_ids=[]
            for n in parts:
                try:
                    if n['Part']=='Beam':
                        num_ids.append(1)
                    elif n['Part']=='Rod':
                        num_ids.append(2)
                    elif n['Part']=='Strut':
                        num_ids.append(3)
                    elif n['Part']=='Joint':
                        num_ids.append(4)
                    elif n['Part']=='JointSupport':
                        num_ids.append(5)
                    elif n['Part']=='RollerSupport':
                        num_ids.append(6)
                    elif n['Part']=='RigidSupport':
                        num_ids.append(7)
                    elif n['Part']=='RigidJointTwo':
                        num_ids.append(8)
                    elif n['Part']=='RigidJointThree':
                        num_ids.append(9)
                    elif n['Part']=='RigidJointFour':
                        num_ids.append(10)
                    elif n['Part']=='NodeDock':
                        num_ids.append(11)
                    elif n['Part']=='Cable':
                        num_ids.append(12)
                    elif n['Part']=='Pulley':
                        num_ids.append(13)
                    elif n['Part']=='Rope':
                        num_ids.append(14)
                    elif n['Part']=='BeamExtender':
                        num_ids.append(15)
                except:
                    pass


            # num_ids = [int(n['Part']) for n in parts]
            # polygons = []
            # for r in a['regions']:
            #
            #     # print(r['shape_attributes'])
            #     # if r['shape_attributes'] is not None:
            #     if r is not None:
            #         polygons.append(r['shape_attributes'])
            # print(polygons)
            #     print(r['shape_attributes'])
            # polygons = [r['shape_attributes'] for r in a['regions']]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])

            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]


            self.add_image(
                "Part",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        info = self.image_info[image_id]
        # print(info)
        if info["source"] != "Part":
            return super(self.__class__, self).load_mask(image_id)

        num_ids = info['num_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            # print("mask.shape, min(mask),max(mask): {}, {},{}".format(mask.shape, np.min(mask),np.max(mask)))
            # print("rr.shape, min(rr),max(rr): {}, {},{}".format(rr.shape, np.min(rr),np.max(rr)))
            # print("cc.shape, min(cc),max(cc): {}, {},{}".format(cc.shape, np.min(cc),np.max(cc)))

            ## Note that this modifies the existing array arr, instead of creating a result array
            ## Ref: https://stackoverflow.com/questions/19666626/replace-all-elements-of-python-numpy-array-that-are-greater-than-some-value
            rr[rr > mask.shape[0]-1] = mask.shape[0]-1
            cc[cc > mask.shape[1]-1] = mask.shape[1]-1

            # print("After fixing the dirt mask, new values:")
            # print("rr.shape, min(rr),max(rr): {}, {},{}".format(rr.shape, np.min(rr),np.max(rr)))
            # print("cc.shape, min(cc),max(cc): {}, {},{}".format(cc.shape, np.min(cc),np.max(cc)))

            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        print(mask)
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)


    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "Part":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    # def load_image(self, image):
    #     # Load image
    #     rgb_image = skimage.io.imread(image)
    #     # If grayscale. Convert to RGB for consistency.
    #     print(rgb_image.ndim)
    #     if rgb_image.ndim != 3:
    #         new_image = skimage.color.gray2rgb(rgb_image)
    #         print(new_image.ndim)
    #         print(new_image.shape)
    #     # If has an alpha channel, remove it for consistency
    #     if rgb_image.shape[-1] == 4:
    #         new_image = rgb_image[..., :3]
    #     # Save images
    #     skimage.io.imsave('rgb.jpg', new_image)
    #     return new_image


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = AHADataset()
    dataset_train.load_aha(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = AHADataset()
    dataset_val.load_aha(args.dataset, "test")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)

        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect aha parts.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/dataset",
                        help='Directory of the dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/weights",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/logs",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = AHAConfig()
    else:
        class InferenceConfig(AHAConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            mrcnn_utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        # aha = AHADataset()

        # image = aha.load_image(args.image)
        # print(image.shape)
        # image = mrcnn_utils.load_image(args.image)
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
