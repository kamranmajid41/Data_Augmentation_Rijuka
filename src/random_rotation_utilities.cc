#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

void composeExternalMatrix(float yaw,
                           float pitch,
                           float roll,
                           float trans_x,
                           float trans_y,
                           float trans_z,
                           Mat& external_matrix) {
  external_matrix.release();
  external_matrix.create(3, 4, CV_64FC1);

  double sin_yaw = sin((double)yaw * CV_PI / 180);
  double cos_yaw = cos((double)yaw * CV_PI / 180);
  double sin_pitch = sin((double)pitch * CV_PI / 180);
  double cos_pitch = cos((double)pitch * CV_PI / 180);
  double sin_roll = sin((double)roll * CV_PI / 180);
  double cos_roll = cos((double)roll * CV_PI / 180);

  external_matrix.at<double>(0, 0) = cos_pitch * cos_yaw;
  external_matrix.at<double>(0, 1) = -cos_pitch * sin_yaw;
  external_matrix.at<double>(0, 2) = sin_pitch;
  external_matrix.at<double>(1, 0) =
      cos_roll * sin_yaw + sin_roll * sin_pitch * cos_yaw;
  external_matrix.at<double>(1, 1) =
      cos_roll * cos_yaw - sin_roll * sin_pitch * sin_yaw;
  external_matrix.at<double>(1, 2) = -sin_roll * cos_pitch;
  external_matrix.at<double>(2, 0) =
      sin_roll * sin_yaw - cos_roll * sin_pitch * cos_yaw;
  external_matrix.at<double>(2, 1) =
      sin_roll * cos_yaw + cos_roll * sin_pitch * sin_yaw;
  external_matrix.at<double>(2, 2) = cos_roll * cos_pitch;

  external_matrix.at<double>(0, 3) = trans_x;
  external_matrix.at<double>(1, 3) = trans_y;
  external_matrix.at<double>(2, 3) = trans_z;
}
Mat Rect2Mat(const Rect& img_rect) {
  Mat srcCoord(3, 4, CV_64FC1);
  srcCoord.at<double>(0, 0) = img_rect.x;
  srcCoord.at<double>(1, 0) = img_rect.y;
  srcCoord.at<double>(2, 0) = 1;
  srcCoord.at<double>(0, 1) = img_rect.x + img_rect.width;
  srcCoord.at<double>(1, 1) = img_rect.y;
  srcCoord.at<double>(2, 1) = 1;
  srcCoord.at<double>(0, 2) = img_rect.x + img_rect.width;
  srcCoord.at<double>(1, 2) = img_rect.y + img_rect.height;
  srcCoord.at<double>(2, 2) = 1;
  srcCoord.at<double>(0, 3) = img_rect.x;
  srcCoord.at<double>(1, 3) = img_rect.y + img_rect.height;
  srcCoord.at<double>(2, 3) = 1;

  return srcCoord;
}

void CircumTransImgRect(const Size& img_size,
                        const Mat& transM,
                        Rect_<double>& CircumRect) {
  Mat cornersMat = Rect2Mat(Rect(0, 0, img_size.width, img_size.height));
  Mat dstCoord = transM * cornersMat;
  double min_x =
      std::min(dstCoord.at<double>(0, 0) / dstCoord.at<double>(2, 0),
               dstCoord.at<double>(0, 3) / dstCoord.at<double>(2, 3));
  double max_x =
      std::max(dstCoord.at<double>(0, 1) / dstCoord.at<double>(2, 1),
               dstCoord.at<double>(0, 2) / dstCoord.at<double>(2, 2));
  double min_y =
      std::min(dstCoord.at<double>(1, 0) / dstCoord.at<double>(2, 0),
               dstCoord.at<double>(1, 1) / dstCoord.at<double>(2, 1));
  double max_y =
      std::max(dstCoord.at<double>(1, 2) / dstCoord.at<double>(2, 2),
               dstCoord.at<double>(1, 3) / dstCoord.at<double>(2, 3));

  CircumRect.x = min_x;
  CircumRect.y = min_y;
  CircumRect.width = max_x - min_x;
  CircumRect.height = max_y - min_y;
}

void CreateMap(const Size& src_size,
               const Rect_<double>& dst_rect,
               const Mat& transMat,
               Mat& map_x,
               Mat& map_y) {
  map_x.create(dst_rect.size(), CV_32FC1);
  map_y.create(dst_rect.size(), CV_32FC1);

  double Z = transMat.at<double>(2, 3);

  Mat invTransMat = transMat.inv();
  Mat dst_pos(3, 1, CV_64FC1);
  dst_pos.at<double>(2, 0) = Z;
  for (int dy = 0; dy < map_x.rows; dy++) {
    dst_pos.at<double>(1, 0) = dst_rect.y + dy;
    for (int dx = 0; dx < map_x.cols; dx++) {
      dst_pos.at<double>(0, 0) = dst_rect.x + dx;
      Mat rMat = -invTransMat(Rect(3, 2, 1, 1)) /
                 (invTransMat(Rect(0, 2, 3, 1)) * dst_pos);
      Mat src_pos = invTransMat(Rect(0, 0, 3, 2)) * dst_pos * rMat +
                    invTransMat(Rect(3, 0, 1, 2));
      map_x.at<float>(dy, dx) =
          src_pos.at<double>(0, 0) + (float)src_size.width / 2;
      map_y.at<float>(dy, dx) =
          src_pos.at<double>(1, 0) + (float)src_size.height / 2;
    }
  }
}

void RotateImage(const Mat& src,
                 Mat& dst,
                 float yaw,
                 float pitch,
                 float roll,
                 float Z = 1000,
                 int interpolation = INTER_LINEAR,
                 int border_mode = BORDER_CONSTANT,
                 const Scalar& border_color = Scalar(0, 0, 0)) {
  // rotation matrix
  Mat rotMat_3x4;
  composeExternalMatrix(yaw, pitch, roll, 0, 0, Z, rotMat_3x4);

  Mat rotMat = Mat::eye(4, 4, rotMat_3x4.type());
  rotMat_3x4.copyTo(rotMat(Rect(0, 0, 4, 3)));

  // From 2D coordinates to 3D coordinates
  // The center of image is (0,0,0)
  Mat invPerspMat = Mat::zeros(4, 3, CV_64FC1);
  invPerspMat.at<double>(0, 0) = 1;
  invPerspMat.at<double>(1, 1) = 1;
  invPerspMat.at<double>(3, 2) = 1;
  invPerspMat.at<double>(0, 2) = -(double)src.cols / 2;
  invPerspMat.at<double>(1, 2) = -(double)src.rows / 2;

  Mat perspMat = Mat::zeros(3, 4, CV_64FC1);
  perspMat.at<double>(0, 0) = Z;
  perspMat.at<double>(1, 1) = Z;
  perspMat.at<double>(2, 2) = 1;

  Mat transMat = perspMat * rotMat * invPerspMat;
  Rect_<double> CircumRect;
  CircumTransImgRect(src.size(), transMat, CircumRect);

  Mat map_x, map_y;
  CreateMap(src.size(), CircumRect, rotMat, map_x, map_y);
  remap(src, dst, map_x, map_y, interpolation, border_mode, border_color);
}

// Keep center and expand rectangle for rotation
Rect ExpandRectForRotate(const Rect& area) {
  Rect exp_rect;

  int w = cvRound(
      std::sqrt((double)(area.width * area.width + area.height * area.height)));

  exp_rect.width = w;
  exp_rect.height = w;
  exp_rect.x = area.x - (exp_rect.width - area.width) / 2;
  exp_rect.y = area.y - (exp_rect.height - area.height) / 2;

  return exp_rect;
}
