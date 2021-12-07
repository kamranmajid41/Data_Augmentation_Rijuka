#ifndef __UTILITIES__
#define __UTILITIES__

#include <opencv2/core/core.hpp>

using namespace cv;

Rect TruncateRect(const Rect& obj_rect, const Size& img_size);
Rect TruncateRectKeepCenter(const Rect& obj_rect, const Size& max_size);
bool LoadAnnotationFile(const std::string& gt_file,
                        std::vector<std::string>& imgpathlist,
                        std::vector<std::vector<Rect>>& rectlist);
bool AddAnnotationLine(const std::string& anno_file,
                       const std::string& img_file,
                       const std::vector<Rect>& obj_rects,
                       const std::string& sep);
bool ReadImageFilesInDirectory(const std::string& img_dir,
                               std::vector<std::string>& image_lists);
bool HasImageExtention(const std::string& filename);
bool ReadCSVFile(
    const std::string& input_file,
    std::vector<std::vector<std::string>>& output_strings,
    const std::vector<std::string>& separater_vec = std::vector<std::string>());
;
std::vector<std::string> TokenizeString(
    const std::string& input_string,
    const std::vector<std::string>& separater_vec);

#endif