#include <functional>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "augmentations.hpp"
#include "data_loader.hpp"

using namespace std;
using namespace cv;

int main() {
  // Load in dataset
  DataLoader dataset(/* YOUR INPUT IMAGE DIRECTORY PATH */);
  RNG rng();

  // Add augmentations
  dataset.AddAugmentation(
      [&rng](const Mat& img) { return RandomHorizontalFlip(img, 0.5, rng); });
  dataset.AddAugmentation(
      [&rng](const Mat& img) { return RandomVerticalFlip(img, 0.5, rng); });
  dataset.AddAugmentation(
      [&rng](const Mat& img) { return RandomSlide(img, 0.5, rng); });

  // Save augmented images on the fly
  dataset.AugmentAndSaveToDirectory(/* YOUR OUTPUT IMAGE DIRECTORY PATH */);

  return 0;
}
