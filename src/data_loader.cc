#include "data_loader.hpp"

#include <iostream>

DataLoader::DataLoader(const std::string& path) { directory_path_ = path; }

void DataLoader::LoadInMemory() {
  for (auto image_path :
       boost::make_iterator_range(directory_iterator(directory_path_), {})) {
    std::string filename = image_path.path().string();
    size_t pos = filename.find_last_of('/');
    if (filename.at(pos + 1) == '.') {
      continue;
    }
    images_.push_back(imread(filename));
  }
  in_memory_ = true;
}

Mat DataLoader::LoadImage(const std::string& path) {
  Mat img = imread(path, 3);
  return img;
}

void DataLoader::AddAugmentation(std::function<Mat(const Mat&)> aug) {
  augmentations_.push_back(aug);
}

void DataLoader::PerformAugmentations() {
  if (in_memory_) {
    for (auto aug : augmentations_) {
      for (Mat& img : images_) {
        img = aug(img);
      }
    }
  } else {
    throw std::runtime_error("Must load in memory to perform augmentations");
  }
}

void DataLoader::AugmentAndSaveToDirectory(const std::string& save_path) {
  create_directories(save_path);
  for (auto image_path :
       boost::make_iterator_range(directory_iterator(directory_path_), {})) {
    std::string filename = image_path.path().string();
    size_t pos = filename.find_last_of('/');
    if (filename.at(pos + 1) == '.') {
      continue;
    }
    Mat img = imread(image_path.path().string());
    filename = save_path + filename.substr(pos, filename.size() - pos);
    for (auto aug : augmentations_) {
      img = aug(img);
    }
    imwrite(filename, img);
  }
}

void DataLoader::SaveImagesToDirectory(const std::string& save_path) {
  if (!in_memory_) {
    throw std::runtime_error("Must load in memory first");
  }
  create_directories(save_path);
  auto img_ptr = images_.begin();
  for (auto image_path :
       boost::make_iterator_range(directory_iterator(directory_path_), {})) {
    std::string filename = image_path.path().string();
    size_t pos = filename.find_last_of('/');
    if (filename.at(pos + 1) == '.') {
      continue;
    }
    filename = save_path + filename.substr(pos, filename.size() - pos);
    imwrite(filename, *img_ptr);
    ++img_ptr;
  }
}

std::vector<Mat>& DataLoader::GetImages() { return images_; }

std::vector<std::function<Mat(const Mat&)>>& DataLoader::GetAugmentations() {
  return augmentations_;
}