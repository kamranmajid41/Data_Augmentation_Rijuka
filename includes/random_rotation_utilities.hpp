#ifndef __RANDOM_ROTATION_UTILITIES__
#define __RANDOM_ROTATION_UTILITIES__

#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

void composeExternalMatrix(float yaw,
                           float pitch,
                           float roll,
                           float trans_x,
                           float trans_y,
                           float trans_z,
                           Mat& external_matrix);

void CreateMap(const Size& src_size,
               const Rect_<double>& dst_rect,
               const Mat& transMat,
               Mat& map_x,
               Mat& map_y);

void RotateImage(const Mat& src,
                 Mat& dst,
                 float yaw,
                 float pitch,
                 float roll,
                 float Z = 1000,
                 int interpolation = INTER_LINEAR,
                 int border_mode = BORDER_CONSTANT,
                 const Scalar& border_color = Scalar(0, 0, 0));

// Keep center and expand rectangle for rotation
Rect ExpandRectForRotate(const Rect& area);

#endif