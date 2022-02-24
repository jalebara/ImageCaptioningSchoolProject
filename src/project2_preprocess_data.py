""" Generates features to use for transformer model
For this project, we leverage a pretrained semantic segmentation model to 
identify regions of interest and extract features from those regions. This allows us 
to effectively tokenize an image and apply mach ine translation techniques
to perform the captioning. 
"""


def generate_cnn_features():
    raise NotImplementedError


def generate_regional_features():
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
