# Python standard library
import os
import time
from pathlib import Path
# Third-party libraries, installable via pip
import numpy as np
## import matplotlib.pyplot as plt  # for threshold mask
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
## from skimage import filters, exposure  # for threshold mask
from tifffile import imread, imwrite
from joblib import dump

# Set input/output behavior
input_dir = Path('./1_human_annotations')
output_dir = Path('./2_random_forest_annotations')
debug_output_dir = Path('./random_forest_intermediate_images')
clf_dir = Path('./classifiers')
clf_dir.mkdir(exist_ok=True)
probability_threshold = 0.6 # Only annotate where you're confident
save_debug_imgs = True

def calculate_features(data):
    assert len(data.shape) == 3 and data.shape[0] > 1 # N-channel, 2D images
    features = []
    for channel in range(data.shape[0] - 1): # Ignore the labels channel
        im = data[channel, :, :].astype('float32')
        # If you know something about your image, and would like to provide a
        # mask as an additional feature, you can define the mask threshold here
        # I would recommend starting with a visualization of a couple test
        # thresholds by uncommenting the next two lines
##        fig, ax = filters.try_all_threshold(im, figsize=(10, 6), verbose=False)
##        plt.show()

        # Define a threshold to generate a mask
##        thresh = filters.threshold_triangle(im)

        # If you'd like to visualize where the threshold is relative to
        # all the pixel values in the image, uncomment the lines below
##        hist, bins_center = exposure.histogram(im)
##        plt.figure()
##        plt.plot(bins_center, hist, lw=2)
##        plt.axvline(thresh, color='k', ls='--')
##        plt.show()

        # NB our features are chosen in a subjective and unprincipled way:
        features.extend(( # These could be calculated more efficiently...
            im,
##            (im > thresh),  # threshold mask
            ndi.gaussian_filter(im, sigma=1),
            ndi.gaussian_filter(im, sigma=2),
            ndi.gaussian_filter(im, sigma=4),
            ndi.gaussian_gradient_magnitude(im, sigma=1),
            ndi.gaussian_gradient_magnitude(im, sigma=2),
            ndi.gaussian_gradient_magnitude(im, sigma=4),
            ndi.sobel(ndi.gaussian_filter(im, sigma=1), axis=-1),
            ndi.sobel(ndi.gaussian_filter(im, sigma=2), axis=-1),
            ndi.sobel(ndi.gaussian_filter(im, sigma=4), axis=-1),
            ndi.sobel(ndi.gaussian_filter(im, sigma=1), axis=-2),
            ndi.sobel(ndi.gaussian_filter(im, sigma=2), axis=-2),
            ndi.sobel(ndi.gaussian_filter(im, sigma=4), axis=-2),
            ndi.gaussian_laplace(im, sigma=1),
            ndi.gaussian_laplace(im, sigma=2),
            ndi.gaussian_laplace(im, sigma=4),
            ndi.convolve(ndi.gaussian_filter(im, sigma=1),
                         weights=((-1, 0,  0),
                                  ( 0, 2,  0),
                                  ( 0, 0, -1))),
            ndi.convolve(ndi.gaussian_filter(im, sigma=2),
                         weights=((-1, 0,  0),
                                  ( 0, 2,  0),
                                  ( 0, 0, -1))),
            ndi.convolve(ndi.gaussian_filter(im, sigma=1),
                         weights=(( 0, 0, -1),
                                  ( 0, 2,  0),
                                  (-1, 0,  0))),
            ndi.convolve(ndi.gaussian_filter(im, sigma=2),
                         weights=(( 0, 0, -1),
                                  ( 0, 2,  0),
                                  (-1, 0,  0))),
            # For larger objects, increase the kernel size (e.g. sigma) used to
            # calculate features. Will increase computation time but improve
            # results. Uncomment lines below
##            ndi.gaussian_filter(im, sigma=4),
##            ndi.gaussian_gradient_magnitude(im, sigma=6),
##            ndi.sobel(ndi.gaussian_filter(im, sigma=4), axis=-1),
##            ndi.sobel(ndi.gaussian_filter(im, sigma=4), axis=-2),
##            ndi.gaussian_laplace(im, sigma=6),
##            ndi.gaussian_laplace(im, sigma=8),
##            ndi.convolve(ndi.gaussian_filter(im, sigma=6),
##             weights=(( 0, 0, -1),
##                      ( 0, 2,  0),
##                      (-1, 0,  0))),
##            ndi.convolve(ndi.gaussian_filter(im, sigma=8),
##             weights=(( 0, 0, -1),
##                      ( 0, 2,  0),
##                      (-1, 0,  0)))
            ))
    return np.stack(features, axis=2) # Not my usual byteorder

def train():
    # Sanity checks on input/output
    assert input_dir.is_dir()
    output_dir.mkdir(exist_ok=True)
    if save_debug_imgs:
        debug_output_dir.mkdir(exist_ok=True)
    input_filenames = [x for x in input_dir.iterdir() if x.suffix == '.tif']
    assert len(input_filenames) > 0, "No annotated images to process"

    print("Loading images and calculating 'features' to train our",
          "random forest model...")
    flattened_features, flattened_labels = [], []
    for fn in input_filenames:
        data = imread(str(fn))
        features = calculate_features(data)
        labels = data[-1, :, :].astype('uint32') # Classification not regression
        if save_debug_imgs: # Inspection is the path to insight
            x = np.moveaxis(features, -1, 0) # Convert to my usual byteorder
            ranges = []
            for ch in x: # Set display range for each channel
                ranges.extend((ch.min(), ch.max()))
            imwrite(debug_output_dir / (fn.stem + '_features.tif'), x,
                    imagej=True, ijmetadata={'Ranges': tuple(ranges)})
        # Only keep pixels that have labels:
        labels_exist = labels.ravel() != 0
        flattened_features.append(features.reshape(-1, features.shape[-1]
                                                   )[labels_exist])
        flattened_labels.append(labels.ravel()[labels_exist])
    # This step is a potential memory-hog:
    flattened_features = np.concatenate(flattened_features)
    flattened_labels = np.concatenate(flattened_labels)
    if len(np.unique(flattened_labels)) < 2:
        print('*'*10, "WARNING", '*'*10)
        print("Only", len(np.unique(flattened_labels)),
              "unique label(s) are currently annotated.")
        print("Useful random forest classification needs two or more labels.")
        print('*'*29)
    print("Done calculating.\n")

    print("Training a random forest classifier with",
          flattened_features.shape[0], "annotated pixels\nand",
          flattened_features.shape[1], "features calculated per pixel... ")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf = clf.fit(flattened_features, flattened_labels)
    print("Done training.\n")

    # save the random forest model so we can use it later
    clf_filename = clf_dir / 'rf_classifier.joblib'
    dump(clf, clf_filename)
    print("Random forest model saved.")

    return clf, input_filenames


def predict(clf, input_filenames):
    # Splitting off the prediction portion so it can be called separately for
    # automated image classification

    print("Loading images, calculating 'features', and saving predictions.",
          end='')
    # input_filenames = [x for x in input_dir.iterdir() if x.suffix == '.tif']
    assert len(input_filenames) > 0, "No annotated images to process"

    for fn in input_filenames:
        print('.', end='')
        data = imread(str(fn))
        features = calculate_features(data)
        probabilities = clf.predict_proba(
            features.reshape(-1, features.shape[-1])
            ).reshape(*data.shape[1:], clf.n_classes_)
        predictions = clf.classes_[np.argmax(probabilities, axis=-1)]
        confident = probabilities.max(axis=-1) > probability_threshold
        unannotated = data[-1, :, :] == 0
        new_annotations = confident & unannotated
        data[-1, :, :][new_annotations] = predictions[new_annotations]
        # Save as tif with per-channel display ranges:
        ranges = []
        for ch in data:
            ranges.extend((ch.min(), ch.max()))
        imwrite(output_dir / fn.name,
                data,
                imagej=True,
                ijmetadata={'Ranges': tuple(ranges)})
    print("\nDone calculating and saving.\n")

print("Waiting for human annotated images...")
last_mtimes = []
while True: # Process images every time input dir images change
    mtimes = [os.stat(x).st_mtime for x in input_dir.iterdir()
              if x.suffix == '.tif']
    if mtimes != last_mtimes:
        if last_mtimes != None:
            print("New images detected in the input directory!\n")
        clf, input_filenames = train()
        predict(clf, input_filenames)
        last_mtimes = mtimes
        print("Waiting for modified input images in", input_dir, "...")
        print("Press 'r' to reload annotations")
    time.sleep(0.1)
