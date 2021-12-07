#include <functional>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "augmentations.hpp"
#include "data_loader.hpp"

using namespace std;
using namespace cv;

int main() {
  // Load in dataset
  DataLoader dataset("/home/vagrant/src/final-project-rijuka/sampleinputs");
  RNG rng = RNG();

  // Add augmentations
  dataset.AddAugmentation(
      [&rng](const Mat& img) { return RandomHorizontalFlip(img, 0.5, rng); });
  dataset.AddAugmentation(
      [&rng](const Mat& img) { return RandomVerticalFlip(img, 0.5, rng); });
  dataset.AddAugmentation(
      [&rng](const Mat& img) { return RandomSlide(img, 0.5, rng); });
  dataset.AddAugmentation([&rng](const Mat& img) {
    return RandomDeform(
        img, {0.01, 0.05}, {0.01, 0.05}, {0.2, 0.4}, {0.2, 0.4}, rng);
  });
  dataset.AddAugmentation(
      [&rng](const Mat& img) { return RandomSlide(img, 1, rng); });
  dataset.AddAugmentation([](const Mat& img) {
    int kernel_size = 3;
    Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) /
                 (float)(kernel_size * kernel_size);
    return Blur(img, kernel);
  });
  dataset.AddAugmentation([&rng](const Mat& img) {
    return RandomNoise(img, {10, 12, 34}, {8, 16, 24}, rng);
  });
  dataset.AddAugmentation([&rng](const Mat& img) {
    double yaw = 15;
    double pitch = 15;
    double roll = 15;
    return RandomRotateImage(img, yaw, pitch, roll, rng);
  });

  // Save augmented images on the fly
  dataset.AugmentAndSaveToDirectory(
      "/home/vagrant/src/final-project-rijuka/sampleoutputs");

  // Save augmented images by loading into memory first
  //   dataset.LoadInMemory();
  //   dataset.PerformAugmentations();
  //   dataset.SaveImagesToDirectory(
  //       "/home/vagrant/src/final-project-rijuka/sampleoutputs");

  return 0;
}