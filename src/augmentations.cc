#include "augmentations.hpp"

#include <boost/filesystem/path.hpp>
#include <cmath>
#include <iostream>

#include "random_rotation_utilities.hpp"
#include "utilities.hpp"

using namespace cv;

/*
  HorizontalFlip

  Flips an image horizontally

  @param const cv::Mat& img -> the original image.

  @return cv::Mat -> adjusted image
*/
Mat HorizontalFlip(const Mat& img) {
  int num_cols = img.cols;
  int num_rows = img.rows;
  Mat dst(num_rows, num_cols, img.type());

  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      dst.at<Vec3b>(i, j) = img.at<Vec3b>(i, num_cols - j - 1);
    }
  }

  return dst;
}

/*
  RandomHorizontalFlip

  Randomly flips an image horizontally

  @param const cv::Mat& img -> the original image
  @param double hflip_ratio -> the probability of flipping
  @param RNG& rng -> opencv RNG object for generating a random flip.

  @return cv::Mat -> adjusted image
*/
Mat RandomHorizontalFlip(const Mat& img, double hflip_ratio, RNG& rng) {
  assert(img.type() == CV_8UC1 || img.type() == CV_8UC3);

  // Random horizontal flip
  Mat dst;
  double flip_prob = rng.uniform(0.0, 1.0);
  if (hflip_ratio > flip_prob) {
    dst = HorizontalFlip(img);
  } else {
    dst = img;
  }
  return dst;
}

/*
  VerticalFlip

  Flips an image vertically

  @param const cv::Mat& img -> the original image.

  @return cv::Mat -> adjusted image
*/
Mat VerticalFlip(const Mat& img) {
  int num_cols = img.cols;
  int num_rows = img.rows;
  Mat dst(num_rows, num_cols, img.type());

  for (int j = 0; j < num_cols; ++j) {
    for (int i = 0; i < num_rows; ++i) {
      dst.at<Vec3b>(i, j) = img.at<Vec3b>(num_rows - i - 1, j);
    }
  }

  return dst;
}

/*
  RandomVerticalFlip

  Randomly flips an image vertically

  @param const cv::Mat& img -> the original image
  @param double hflip_ratio -> the probability of flipping
  @param RNG& rng -> opencv RNG object for generating a random flip.

  @return cv::Mat -> adjusted image
*/
Mat RandomVerticalFlip(const Mat& img, double vflip_ratio, RNG& rng) {
  assert(img.type() == CV_8UC1 || img.type() == CV_8UC3);

  // Random vertical flip
  Mat dst;
  double flip_prob = rng.uniform(0.0, 1.0);
  if (vflip_ratio > flip_prob) {
    dst = VerticalFlip(img);
  } else {
    dst = img;
  }
  return dst;
}

Mat RandomRotateImage(const Mat& src,
                      double yaw_sigma,
                      double pitch_sigma,
                      double roll_sigma,
                      RNG& rng,
                      const Rect& area,
                      double Z,
                      int interpolation,
                      int border_mode,
                      const Scalar& border_color) {
  double yaw =
      std::min<double>(60, std::max<double>(-60, rng.gaussian(yaw_sigma)));
  double pitch =
      std::min<double>(60, std::max<double>(-60, rng.gaussian(pitch_sigma)));
  double roll =
      std::min<double>(60, std::max<double>(-60, rng.gaussian(roll_sigma)));

  Rect rect = (area.width <= 0 || area.height <= 0)
                  ? Rect(0, 0, src.cols, src.rows)
                  : ExpandRectForRotate(area);
  rect = TruncateRectKeepCenter(rect, src.size());

  Mat rot_img;
  RotateImage(src(rect).clone(),
              rot_img,
              yaw,
              pitch,
              roll,
              Z,
              interpolation,
              border_mode,
              border_color);
  return rot_img;

  Rect dst_area((rot_img.cols - area.width) / 2,
                (rot_img.rows - area.height) / 2,
                area.width,
                area.height);
  dst_area = TruncateRectKeepCenter(dst_area, rot_img.size());
  Mat dst = rot_img(dst_area).clone();
  return dst;
}

/*
  Slide

  translates an image by any positive or negative integer value in both x and y
  directions.

  @param const cv::Mat& img -> the original image
  @param int x_shift -> pos or neg integer value to shift by in x direction
  @param int y_shift -> pos or neg integer value to shift by in y direction

  @return Mat -> adjusted image
*/
Mat Slide(const Mat& img, int x_shift, int y_shift) {
  assert(img.type() == CV_8UC1 || img.type() == CV_8UC3);

  int num_cols = img.cols;
  int num_rows = img.rows;

  while (x_shift < 0) x_shift += num_cols;
  while (y_shift < 0) y_shift += num_rows;

  Mat to_return(num_rows, num_cols, CV_8UC3);

  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      to_return.at<Vec3b>((i + y_shift) % num_rows, (j + x_shift) % num_cols) =
          img.at<Vec3b>(i, j);
    }
  }

  return to_return;
}

/*
  RandomSlide

  randomly slides an image based on a certain slide ratio (% of the time to
  slide the image). randomly generates the amount to slide the image (either
  left or right and either up or down each by a random amount)

  @param const cv::Mat& img -> the original image
  @param double slide_ratio -> the ratio of images that do get slid vs not
                               slid
  @param cv::RNG& rng -> opencv RNG object for generating whether or not to
                         slide and the amount to slide by.

  @return Mat -> adjusted image
*/
Mat RandomSlide(const Mat& img, double slide_ratio, RNG& rng) {
  Mat to_return;
  double slide_prob = rng.uniform(0.0, 1.0);

  if (slide_ratio > slide_prob) {
    int num_cols = img.cols;
    int num_rows = img.rows;
    int x_slide = rng.uniform(-1 * num_cols, num_cols);
    int y_slide = rng.uniform(-1 * num_rows, num_rows);
    to_return = Slide(img, x_slide, y_slide);
  } else {
    to_return = img;
  }

  return to_return;
}

/*
  RandomDeform

  does a warping of the image in the vertical and horizontal directions based
  off sin & cos waves

  @param const cv::Mat& img -> the original image
  @param double x_amp -> horizontal range for distortion amplitude
  @param double y_amp -> vertical range for distortion amplitude
  @param double x_freq -> horizontal range for distortion frequency
  @param double y_freq -> vertical range for distortion frequency
  @param cv::RNG& rng -> opencv RNG object for generating a random deformation.

  @return Mat -> the deformed image
*/

Mat RandomDeform(const Mat& img,
                 std::pair<double, double> x_amp,
                 std::pair<double, double> y_amp,
                 std::pair<double, double> x_freq,
                 std::pair<double, double> y_freq,
                 RNG& rng) {
  int num_cols = img.cols;
  int num_rows = img.rows;
  Mat to_return(num_rows, num_cols, CV_8UC3, Scalar(0, 0, 0));

  int x_wave_amp = rng.uniform(x_amp.first * num_rows, x_amp.second * num_rows);
  int x_wave_freq =
      rng.uniform(x_freq.first * num_rows, x_freq.second * num_rows);
  int y_wave_amp = rng.uniform(y_amp.first * num_cols, y_amp.second * num_cols);
  int y_wave_freq =
      rng.uniform(y_freq.first * num_cols, y_freq.second * num_cols);

  for (int i = 0; i < num_rows; i++) {
    for (int j = 0; j < num_cols; j++) {
      int x_offset =
          std::round(x_wave_amp * std::sin((2 * M_PI * i) / x_wave_freq));
      int y_offset =
          std::round(y_wave_amp * std::cos((2 * M_PI * i) / y_wave_freq));
      if (i + y_offset < num_rows && j + x_offset < num_cols) {
        to_return.at<Vec3b>(i, j) =
            img.at<Vec3b>((i + y_offset) % num_rows, (j + x_offset) % num_cols);
      }
    }
  }

  return to_return;
}

/*
  Blur

  Blurs the source image according to a kernel

  @param const Mat& src -> the original image
  @param const Mat& kernel -> the kernel used to blur the source
  @param const Point& anchor -> the position of the anchor relative to the
  kernel
  @param double delta -> a value to be added to the pixels during the blur
  process
  @param int depth -> the depth of the destination image.

  @return Mat -> the blurred image
*/

Mat Blur(const Mat& src,
         const Mat& kernel,
         const Point& anchor,
         double delta,
         int depth) {
  Mat dst;
  filter2D(src, dst, depth, kernel, anchor, delta, BORDER_DEFAULT);
  return dst;
}

/*
  RandomNoise

  Adds random noise to the source image according to a gaussian distribution

  @param const Mat& src -> the original image
  @param const std::vector<double>& mean -> the mean of the distribution for
  each channel
  @param const std::vector<double>& std_dev -> the standard deviation of the
  distribution for each channel
  @param RNG& rng -> opencv RNG object for generating a random number.

  @return Mat -> the image with noise
*/

Mat RandomNoise(const Mat& src,
                const std::vector<double>& mean,
                const std::vector<double>& std_dev,
                RNG& rng) {
  Mat dst = src.clone();
  int num_cols = src.cols;
  int num_rows = src.rows;

  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      for (int k = 0; k < mean.size(); ++k) {
        double noise = rng.gaussian(std_dev.at(k)) + mean.at(k);
        noise = std::max<double>(0, noise + dst.at<Vec3b>(i, j)[k]);
        noise = std::min<double>(255, noise);
        dst.at<Vec3b>(i, j)[k] = noise;
      }
    }
  }
  return dst;
}
