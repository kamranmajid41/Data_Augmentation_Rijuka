#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <functional>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cv;
using namespace boost::filesystem;

class DataLoader {
public:
  DataLoader();
  DataLoader(const std::string& path);
  void LoadInMemory();
  void AddAugmentation(std::function<Mat(const Mat&)> aug);
  void PerformAugmentations();
  void AugmentAndSaveToDirectory(const std::string& save_path);
  void SaveImagesToDirectory(const std::string& path);
  std::vector<Mat>& GetImages();
  std::vector<std::function<Mat(const Mat&)>>& GetAugmentations();

private:
  Mat LoadImage(const std::string& path);
  std::string directory_path_;
  std::vector<Mat> images_;
  bool in_memory_ = false;
  std::vector<std::function<Mat(const Mat&)>> augmentations_;
};

#endif
