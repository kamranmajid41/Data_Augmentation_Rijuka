#include "utilities.hpp"

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>
#include <fstream>

using namespace cv;

Rect TruncateRect(const Rect& obj_rect, const Size& img_size) {
  Rect resize_rect = obj_rect;
  if (obj_rect.x < 0) {
    resize_rect.x = 0;
    resize_rect.width += obj_rect.x;
  }
  if (obj_rect.y < 0) {
    resize_rect.y = 0;
    resize_rect.height += obj_rect.y;
  }
  if (resize_rect.x + resize_rect.width > img_size.width) {
    resize_rect.width = img_size.width - resize_rect.x;
  }
  if (resize_rect.y + resize_rect.height > img_size.height) {
    resize_rect.height = img_size.height - resize_rect.y;
  }

  return resize_rect;
}

Rect TruncateRectKeepCenter(const Rect& obj_rect, const Size& max_size) {
  Rect exp_rect = obj_rect;
  if (exp_rect.x < 0) {
    exp_rect.width += 2 * exp_rect.x;
    exp_rect.x = 0;
  }
  if (exp_rect.y < 0) {
    exp_rect.height += 2 * exp_rect.y;
    exp_rect.y = 0;
  }
  if (exp_rect.x + exp_rect.width > max_size.width) {
    exp_rect.x += (exp_rect.x + exp_rect.width - max_size.width) / 2;
    exp_rect.width = max_size.width - exp_rect.x;
  }
  if (exp_rect.y + exp_rect.height > max_size.height) {
    exp_rect.y += (exp_rect.y + exp_rect.height - max_size.height) / 2;
    exp_rect.height = max_size.height - exp_rect.y;
  }
  return exp_rect;
}
