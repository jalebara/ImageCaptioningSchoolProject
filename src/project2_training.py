""" Performs the main training for the meshed memory transformer.
There are two phases of training for the MMT. First, we perform supervised 
training using negative log likelihood (Cross Entropy Loss).The second phase 
of training involves fine tuning using Reinforement Learning. Due to the decorrelation 
of the loss and the natural languange metrics in convnetional deep learning training 
methods, it is necessary to directly involve the desired metrics through 
reinforcement learning.
"""


def parse_args():
    raise NotImplementedError


def train_mmt_cross_entropy():
    raise NotImplementedError


def train_mmt_rl():
    raise NotImplementedError


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
