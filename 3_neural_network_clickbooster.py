# Python standard library
from pathlib import Path
from random import shuffle
# Third-party libraries, installable via pip
import numpy as np
from tifffile import imread, imwrite
# Install instructions: https://pytorch.org/get-started/locally/#start-locally
# Make sure to install pytorch with CUDA support, and have a CUDA-able GPU.
print('Importing pytorch...', end='')
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

# Check if we have a compatible GPU available
use_GPU = False
if torch.cuda.is_available():
    print('\nCUDA-compatible GPU available for pytorch.')
    import torch.cuda as cuda
    use_GPU = True
else:
    print('\nNo GPU available for pytorch.')
    use_GPU = False

# Test for torch import
torch.manual_seed(27)
x = torch.rand(5, 3)
print('\n', x, '\n Import finished.')

# Set input/output behavior
input_dir = Path('./2_random_forest_annotations')
##input_dir = Path('./1_human_annotations')
input_features_dir = Path('./random_forest_intermediate_images') # Temporary
output_dir = Path('./3_neural_network_annotations')
other_output_dir = Path('./neural_network_intermediate_files')
saved_state_path = other_output_dir / 'neural_network_state.pt'
save_debug_imgs = True
max_label = 3

# Sanity checks on input/output
assert input_dir.is_dir(), "Input directory does not exist"
assert input_features_dir.is_dir(), "Input features directory does not exist"
img_filenames = [x for x in input_dir.iterdir() if x.suffix == '.tif']
assert len(img_filenames) > 0, "No annotated images to process"
img_features_filenames = [input_features_dir / (x.stem + '_features.tif')
                          for x in img_filenames]
for im in img_features_filenames:
    assert im.is_file(), "Missing features: %s"%im
example_image = imread(str(img_filenames[0]))
num_input_channels = example_image.shape[0] - 1
assert all(label in range(max_label + 1)
           for label in np.unique(example_image[-1, :, :]))
assert max_label < 2**8 # Bro you don't need more
example_features = imread(str(img_features_filenames[0]))
num_features = example_features.shape[0]
output_dir.mkdir(exist_ok=True)
if save_debug_imgs:
    other_output_dir.mkdir(exist_ok=True)

# nn.Linear expects the wrong data shape, whereas nn.Conv2d expects the correct
# data shape. Given a kernel size of (1, 1), a Conv2d layer does the same thing
# as a Linear layer. We've wrapped it here for clarity
def linear_layer_custom(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, (1, 1))


# Pick a neural network to train.
print("Initialzing model...", end='')
class ManualFeatureNet(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.classification_stack = nn.Sequential(
            nn.BatchNorm2d(num_features),
            linear_layer_custom(num_features, 2*num_features),
            nn.ReLU(),
            nn.BatchNorm2d(num_features*2),
            linear_layer_custom(2*num_features, 4*num_features),
            nn.ReLU(),
            nn.BatchNorm2d(num_features*4),
            linear_layer_custom(4*num_features, 8*num_features),
            nn.ReLU(),
            nn.BatchNorm2d(num_features*8),
            linear_layer_custom(8*num_features, num_classes),
            nn.ReLU()
            )

    def forward(self, x):
        return self.classification_stack(x)

class AutoFeatureNet(nn.Module):
    def __init__(self, num_input_channels, num_classes):
        super().__init__()
        ##        # Convolution feature map stack to generate a low-dimensional
##        # representation of the input images
##        self.feature_map_pooling_stack = nn.Sequential(
##            # Layer block A
##            nn.Conv2d(num_input_channels, 32, (4, 4), padding='same'),
##            nn.BatchNorm2d(32),
##            nn.ReLU(),
##            nn.MaxPool2d(2, stride=2),
##            # layers block B
##            nn.Conv2d(32, 64, (3, 3), padding='same'),
##            nn.BatchNorm2d(64),
##            nn.ReLU(),
##            # Layers block C
##            nn.Conv2d(64, 64, (3, 3), padding='same'),
##            nn.BatchNorm2d(64),
##            nn.ReLU(),
##            nn.MaxPool2d(2, stride=2),
##            # Layers block D
##            nn.Conv2d(64, 128, (3, 3), padding='same'),
##            nn.BatchNorm2d(128),
##            nn.ReLU(),
##            # Layers block E
##            nn.Conv2d(128, 200, (3, 3), padding='same'),
##            nn.BatchNorm2d(200),
##            nn.ReLU()
##            )

        # Convolution feature map stack to generate feature maps
        # Adjust the kernel size according to the size of object(s) you
        # wish to annotate (larger object:larger kernel:longer run time)
        kernel_size = 7  # use odd numbers to prevent padding warning
        self.feature_map_conv_stack = nn.Sequential(
            # Layer block A
            nn.Conv2d(num_input_channels, 32, (kernel_size, kernel_size),
                      padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # layers block B
            nn.Conv2d(32, 64, (kernel_size, kernel_size), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Layers block C
            nn.Conv2d(64, 64, (kernel_size, kernel_size), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Layers block D
            nn.Conv2d(64, 128, (kernel_size, kernel_size), padding='same'),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Layers block E
            nn.Conv2d(128, 200, (kernel_size, kernel_size), padding='same'),
            nn.BatchNorm2d(200),
            nn.ReLU()
            )
        # Classification stack designed to be paired with the auto feature map
        # stack
        self.feature_map_class_stack = nn.Sequential(
            nn.BatchNorm2d(200),
            linear_layer_custom(200, 200),
            nn.ReLU(),
            nn.BatchNorm2d(200),
            linear_layer_custom(200, num_classes),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.feature_map_conv_stack(x)
##        x = self.feature_map_pooling_stack(x)
        return self.feature_map_class_stack(x)

model = AutoFeatureNet(num_input_channels=num_input_channels,
                       num_classes=max_label)

if use_GPU:
    model = model.cuda()
print(" done.")

# NB learning rate 1e-3 and weight decay 1e-4 might be hilariously
# wrong. Be careful! Expect to tune both of these parameters by 1-2
# orders of magnitude to find a good balance of obedience and
# independence. There's probably a more intelligent approach.
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-4, amsgrad=True)

# Pick up where we left off, if we've already done some training:
starting_epoch = 0
if saved_state_path.is_file():
    print("Loading saved model and optimizer state...", end='')
    try:
        checkpoint = torch.load(saved_state_path)
    except RuntimeError:
        print("\nSaved state may be corrupted; loading backup...", end='')
        checkpoint = torch.load(saved_state_path.parent /
                                (saved_state_path.stem + '_backup.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    starting_epoch = 1 + checkpoint['epoch']
    model.train()
    print(' done.')

def loss_fn(output, target):
    # Our target has an extra slice at the beginning for unannotated pixels:
    target = target[:, 1:, :, :]
    assert output.shape == target.shape
    annotated_pixels = torch.sum(target)
    probs = F.softmax(output, dim=1)
    return torch.sum(probs * target) / -annotated_pixels
##    # Each class gets an equal vote, and each annotated pixel in a given
##    # class gets an equal vote:
##    annotated_pixels_per_label = torch.sum(target, dim=(2, 3), keepdim=True)
##    label_weight = 1e-9 + max_label * annotated_pixels_per_label
##    probs = F.softmax(output, dim=1)
##    return torch.sum(probs * target / (-label_weight))

def load_data(annotations_path, features_path):
    annotations = imread(str(annotations_path))
    features = imread(str(features_path))
    assert annotations.shape[0] == 1 + num_input_channels
    assert features.shape[0] == num_features
    assert len(annotations.shape) == 3
    assert len(features.shape) == 3
    assert annotations.shape[1:] == features.shape[1:]
    # Match shape and dtype to what torch expects: (batch_size,
    # num_input_channels, y, x) and float32
    if use_GPU:
        input_ = torch.cuda.FloatTensor(
                    annotations[np.newaxis, :-1, :, :].astype('float32'))
##        input_ = torch.cuda.FloatTensor(
##                    features[np.newaxis, ...].astype('float32'))
    else:
        input_ = torch.FloatTensor(
                    annotations[np.newaxis,  :-1, :, :].astype('float32'))
##        input_ = torch.FloatTensor(
##                    features[np.newaxis, ...].astype('float32'))
    input_.requires_grad = True
    # Last channel holds our annotations of the raw image. Annotation
    # values are ints ranging from 0 to max_label; each different
    # annotation value signals a different label. We unpack these into a
    # "1-hot" representation called 'target'.
    labels = annotations[np.newaxis, -1:, :, :].astype('uint8')
    assert labels.max() <= max_label
    if use_GPU:
        # We pass a small dtype to the GPU, but the on-GPU dtype has to be
        # Long to work with .scatter_():
        labels = torch.cuda.LongTensor(labels)
        # An empty Boolean array to be filled with our "1-hot" representation:
        target = torch.cuda.BoolTensor(
            1, max_label + 1, annotations.shape[1], annotations.shape[2]
            ).zero_() # Initialization to zero is not automatic!
    else:
        labels = torch.LongTensor(labels)
        target = torch.BoolTensor(
            1, max_label + 1, annotations.shape[1], annotations.shape[2]
            ).zero_() # Initialization to zero is not automatic!
    # Copy each class into its own boolean image:
    target.scatter_(dim=1, index=labels.data, value=True)
    return input_, target

def save_output(output, img_path):
    guess = F.softmax(output.cpu().data, dim=1).numpy().astype('float32')
    imwrite(img_path,
            guess,
            photometric='MINISBLACK',
            imagej=True,
            ijmetadata={'Ranges:', (0, 1)*guess.shape[1]})
# for epoch in range(starting_epoch, 100000): # Basically forever
for epoch in range(starting_epoch, 2):
    img_paths = [x for x in input_dir.iterdir() if x.suffix == '.tif']
    img_features_paths = [input_features_dir / (x.stem + '_features.tif')
                              for x in img_paths]
    loss_list = []
    print("\nEpoch", epoch)
    for i, img_path in enumerate(img_paths):
        print('.', sep='', end='')
        input_, target = load_data(img_path, img_features_paths[i])
        output = model(input_)
        loss = loss_fn(output, target)
##        print(img_path, loss.detach().item() != 0)
        if loss.detach().item() != 0: # Don't bother if there's no annotations
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Outputs for inspection
        loss_list.append(loss.detach().item())
        save_output(output, output_dir / img_path.name)
        if save_debug_imgs and i == (len(img_paths) - 1):
            save_output(output,
                        other_output_dir / ('e%06i_'%epoch + img_path.name))
    print('\nLosses:')
    print(''.join('%0.5f '%x for x in loss_list))
    if saved_state_path.is_file():
        saved_state_path.replace(saved_state_path.parent /
                                 (saved_state_path.stem + '_backup.pt'))
    torch.save(
        {'epoch': epoch,
         'model_state_dict': model.state_dict(),
         'optimizer_state_dict': optimizer.state_dict()},
        saved_state_path)
