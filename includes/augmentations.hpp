#ifndef __AUGMENTATIONS__
#define __AUGMENTATIONS__

//#include <opencv/core/core.hpp>
#include <algorithm>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/core.hpp>
#include <utility>
#include <vector>

using namespace cv;

Mat RandomHorizontalFlip(const Mat& img, double hflip_ratio, RNG& rng);
Mat HorizontalFlip(const Mat& img);
Mat RandomVerticalFlip(const Mat& img, double vflip_ratio, RNG& rng);
Mat VerticalFlip(const Mat& img);

Mat RandomRotateImage(const Mat& src,
                      double yaw_range,
                      double pitch_range,
                      double roll_range,
                      RNG& rng,
                      const Rect& area = Rect(-1, -1, 0, 0),
                      double Z = 1000,
                      int interpolation = INTER_LINEAR,
                      int border_mode = BORDER_CONSTANT,
                      const Scalar& border_color = Scalar(0, 0, 0));

Mat Slide(const Mat& img, int x_shift, int y_shift);
Mat RandomSlide(const Mat& img, double slide_ratio, RNG& rng);
Mat RandomDeform(const Mat& img,
                 std::pair<double, double> x_amp,
                 std::pair<double, double> y_amp,
                 std::pair<double, double> x_freq,
                 std::pair<double, double> y_freq,
                 RNG& rng);
Mat Blur(const Mat& src,
         const Mat& kernel,
         const Point& anchor = Point(-1, -1),
         double delta = 0,
         int depth = -1);
Mat RandomNoise(const Mat& src,
                const std::vector<double>& mean,
                const std::vector<double>& variance,
                RNG& rng);

#endif
