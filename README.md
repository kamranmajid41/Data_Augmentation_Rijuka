![alt text](https://github.com/cs128-2021c/final-project-rijuka/blob/master/logo-rijuka.png)

# CS 128 Final Project: Image Augmentation Pipeline

## Inspiration

Augmentation is a technique commonly used in machine learning to increase the size of a dataset and therefore make a learning model more reliable.

## What it does

Our data augmentation pipeline takes in images of various formats and performs random augmentations on them, producing alternative images to train with. 

## How we built it

To build this pipeline, our group used boost, OpenCV, and wrote the augmentations/data loader in C++. 

## Challenges we ran into 

Implementing some of the augmentations required high level linear algebra and statistical knowledge which we had to learn. 

## Accomplishments that we're proud of 

We are proud of the functionality of our project, and our successful implementation of harder augmentations (rotation, noise, etc.).

## What we learned 

Throughout this project, we learned a significant amount about machine learning, working with images, and linear algebra. 

## Potential future improvements

In the future, we could implement even more augmentations and create a user interface to make the process of image augmentation even more streamlined and accessible.

## Try it out yourself!

To use our augmentation pipeline, you first need to install the required dependencies, boost and OpenCV.

    sudo apt update
    sudo apt-get install libboost-all-dev
    sudo apt install libopencv-dev python3-opencv

Afterwards, modify the src/main.cc file to include your image directory path. Add any augmentations, perform the augmentations, and save to an output directory.

    DataLoader dataset(/* YOUR INPUT IMAGE DIRECTORY PATH */);

    /*
    ADD AUGMENTATIONS
    */
    
    dataset.AugmentAndSaveToDirectory(/* YOUR OUTPUT IMAGE DIRECTORY PATH */);

To build and execute src/main.cc, run the following from the Makefile

    make main
    ./bin/main

