from mrcnn import model, visualize, config
from skimage import io
import sys
import matplotlib
if sys.platform == 'linux2' and 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
from glob import glob
import numpy as np

class TrainConfig(config.Config):
    NAME = 'cvppp'
    NUM_CLASSES = 1 + 1 # background + leaf
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 256

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 500

    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    def to_string(self):
        """Compile String Of Configuration values."""
        s = "Configurations:"
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                s = s + "\n" + "{:30} {}".format(a, getattr(self, a))
        s = s + "\n"
        return s

class InferenceConfig(TrainConfig):
    IMAGES_PER_GPU = 1

def arguments():
    """
    CMD args
    """
    parser = argparse.ArgumentParser(description='Performs inference using a Mask RCNN Model')
    parser.add_argument('--dataPattern', type=str, required=True,
                        help="A glob file path pattern in quotations. e.g. '/data/plants/plant????_*_image.png'")
    parser.add_argument('--outputDir', type=str, required=True,
                        help='Directory to save all outputs to')
    parser.add_argument('--weightsPath', type=str, required=True,
                        help='Path to model weights to use (h5 file)')
    parser.add_argument('--verboseDetection', dest='verbose', action='store_true')
    parser.add_argument('--quietDetection', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)

    parser.add_argument('--useCPU', dest='useCPU', action='store_true')
    parser.add_argument('--useGPU', dest='useCPU', action='store_false')
    parser.set_defaults(useCPU=False)
    return parser.parse_args()

def load_image(im_path):
    image = io.imread(im_path)
    # Check for alpha channel or single channel

    if len(image.shape) > 2 and image.shape[2] > 3:
        image = image[:, :, :3]
    elif len(image.shape) == 2:
        print("Converting singel channel image to 3 channel by repeating image")
        image = np.stack((image, image, image), axis=2)

    return image

def mask_to_rgb(mask):
    """
    Converts a mask to RGB Format
    """
    colours = visualize.random_colors(mask.shape[2])
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3))

    for i in range(mask.shape[2]):
        for c in range(3):
            rgb_mask[:, :, c] = np.where(mask[:, :, i] != 0, colours[i][c], rgb_mask[:, :, c])

    return rgb_mask

def inference():
    """
    Main inference loop
    """
    args = arguments()

    # Create output dir
    assert not os.path.isdir(args.outputDir), "output dir already exists"
    os.mkdir(args.outputDir)

    # Init config
    configuration = InferenceConfig()

    if args.useCPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # Init model
    inference_model = model.MaskRCNN(mode="inference",
                          config=configuration,
                          model_dir=args.outputDir)

    assert os.path.exists(args.weightsPath), "Weights file does not exist at " + args.weightsPath
    inference_model.load_weights(args.weightsPath, by_name=True)

    # Predict Images
    with open(os.path.join(args.outputDir, 'leafCounts.csv'), 'a') as count_file:
        count_file.write("Image, Count\n")
        for im_path in glob(args.dataPattern):
            out_path = os.path.join(args.outputDir, os.path.basename(im_path))

            print("Saving prediction for", im_path, "at", out_path)

            image = load_image(im_path)

            results = inference_model.detect([image], verbose=args.verbose)
            rgb_mask = mask_to_rgb(results[0]['masks'])
            io.imsave(out_path, rgb_mask)

            count_file.write(os.path.basename(im_path) + ", " + str(results[0]['masks'].shape[2]) + "\n")

if __name__ == '__main__':
    inference()
