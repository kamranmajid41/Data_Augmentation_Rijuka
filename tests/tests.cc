#include <string>

#include "augmentations.hpp"
#include "catch.hpp"
#include "data_loader.hpp"
#include "utilities.hpp"

using namespace cv;

/*
HELPERS
*/

bool MatsAreEqual(const Mat& a, const Mat& b) {
  // Check if two images are identical
  return countNonZero(a != b) == 0;
}

/*
TEST CASES
*/

TEST_CASE("Check image loading simple", "[image_init]") {
  std::string directory_path =
      "/home/vagrant/src/final-project-rijuka/sampleinputs";
  DataLoader dataset(directory_path);
  dataset.LoadInMemory();
  std::vector<Mat>& images = dataset.GetImages();
  int i = 0;
  for (auto image_path :
       boost::make_iterator_range(directory_iterator(directory_path), {})) {
    std::string filename = image_path.path().string();
    size_t pos = filename.find_last_of('/');
    if (filename.at(pos + 1) == '.') {
      continue;
    }
    Mat test = imread(filename, 3);
    Mat gen = images.at(i);
    REQUIRE(MatsAreEqual(gen, test));
    ++i;
  }
}

TEST_CASE("Slide simple", "[slide]") {
  Mat img_orig = imread(
      "/home/vagrant/src/final-project-rijuka/sampleinputs/tinypix.ppm", 3);
  Mat img_to_slide = imread(
      "/home/vagrant/src/final-project-rijuka/sampleinputs/tinypix.ppm", 3);

  Mat img_slid = Slide(img_to_slide, 2, 2);

  REQUIRE(img_orig.at<Vec3b>(0, 0) == img_slid.at<Vec3b>(2, 2));
  REQUIRE(img_orig.at<Vec3b>(1, 1) == img_slid.at<Vec3b>(3, 3));
  REQUIRE(img_orig.at<Vec3b>(2, 2) == img_slid.at<Vec3b>(0, 0));
  REQUIRE(img_orig.at<Vec3b>(2, 0) == img_slid.at<Vec3b>(0, 2));
}

TEST_CASE("Load augmentations", "[load_augs]") {
  DataLoader dataset("/home/vagrant/src/final-project-rijuka/sampleinputs");
  dataset.AddAugmentation(HorizontalFlip);
  dataset.AddAugmentation(VerticalFlip);
  std::vector<std::function<Mat(const Mat&)>> augmentations =
      dataset.GetAugmentations();
  REQUIRE(augmentations.at(0).target_type().name() ==
          std::function<Mat(const Mat&)>(HorizontalFlip).target_type().name());
  REQUIRE(augmentations.at(1).target_type().name() ==
          std::function<Mat(const Mat&)>(VerticalFlip).target_type().name());
}

TEST_CASE("Check output directories created", "[check_out_dir]") {
  DataLoader dataset("/home/vagrant/src/final-project-rijuka/sampleinputs");
  REQUIRE_THROWS(dataset.PerformAugmentations());

  std::string out_dir = "/home/vagrant/src/final-project-rijuka/test_out";
  dataset.AugmentAndSaveToDirectory(out_dir);
  REQUIRE(boost::filesystem::exists(out_dir));
  boost::filesystem::remove_all(out_dir);

  dataset.LoadInMemory();
  dataset.SaveImagesToDirectory(out_dir);
  REQUIRE(boost::filesystem::exists(out_dir));
  boost::filesystem::remove_all(out_dir);
}

TEST_CASE("Horizontal flip", "[horizontal]") {
  Mat img =
      imread("/home/vagrant/src/final-project-rijuka/sampleinputs/ocean.ppm");
  Mat flipped = HorizontalFlip(img);
  int num_cols = img.cols;
  REQUIRE(flipped.at<Vec3b>(0, num_cols - 1) == img.at<Vec3b>(0, 0));
  REQUIRE(flipped.at<Vec3b>(4, num_cols - 1) == img.at<Vec3b>(4, 0));
  REQUIRE(flipped.at<Vec3b>(8, num_cols - 9) == img.at<Vec3b>(8, 8));
  REQUIRE(flipped.at<Vec3b>(0, num_cols - 9) == img.at<Vec3b>(0, 8));
  REQUIRE(flipped.at<Vec3b>(4, num_cols - 5) == img.at<Vec3b>(4, 4));
  REQUIRE(flipped.at<Vec3b>(8, num_cols - 17) == img.at<Vec3b>(8, 16));
  REQUIRE(flipped.at<Vec3b>(100, num_cols - 5) == img.at<Vec3b>(100, 4));
  REQUIRE(flipped.at<Vec3b>(150, num_cols - 17) == img.at<Vec3b>(150, 16));
  REQUIRE(flipped.at<Vec3b>(250, num_cols - 251) == img.at<Vec3b>(250, 250));
  REQUIRE(flipped.at<Vec3b>(200, num_cols - 401) == img.at<Vec3b>(200, 400));
}

TEST_CASE("Vertical flip", "[vertical]") {
  Mat img =
      imread("/home/vagrant/src/final-project-rijuka/sampleinputs/ocean.ppm");
  Mat flipped = VerticalFlip(img);
  int num_rows = img.rows;
  REQUIRE(flipped.at<Vec3b>(num_rows - 1, 0) == img.at<Vec3b>(0, 0));
  REQUIRE(flipped.at<Vec3b>(num_rows - 5, 0) == img.at<Vec3b>(4, 0));
  REQUIRE(flipped.at<Vec3b>(num_rows - 9, 8) == img.at<Vec3b>(8, 8));
  REQUIRE(flipped.at<Vec3b>(num_rows - 1, 8) == img.at<Vec3b>(0, 8));
  REQUIRE(flipped.at<Vec3b>(num_rows - 5, 4) == img.at<Vec3b>(4, 4));
  REQUIRE(flipped.at<Vec3b>(num_rows - 9, 16) == img.at<Vec3b>(8, 16));
  REQUIRE(flipped.at<Vec3b>(num_rows - 101, 4) == img.at<Vec3b>(100, 4));
  REQUIRE(flipped.at<Vec3b>(num_rows - 151, 16) == img.at<Vec3b>(150, 16));
  REQUIRE(flipped.at<Vec3b>(num_rows - 251, 150) == img.at<Vec3b>(250, 150));
  REQUIRE(flipped.at<Vec3b>(num_rows - 201, 200) == img.at<Vec3b>(200, 200));
}

TEST_CASE("image blur", "[blur]") {
  Mat img =
      imread("/home/vagrant/src/final-project-rijuka/sampleinputs/ocean.ppm");
  int num_cols = img.cols;
  int num_rows = img.rows;
  int kernel_size = 3;
  Mat kernel = Mat::ones(kernel_size, kernel_size, CV_32F) /
               (float)(kernel_size * kernel_size);
  Mat test_blurred = Blur(img, kernel);
  Mat dst;
  filter2D(img, dst, -1, kernel, Point(-1, -1), 0, BORDER_DEFAULT);
  REQUIRE(MatsAreEqual(test_blurred, dst));
}