#include "XReader.h"
#include <iostream>


bool stringToBool(const std::string& str) {
    std::string lowerStr = str;
    // 转换字符串为小写，以便不区分大小写
    std::transform(lowerStr.begin(), lowerStr.end(), lowerStr.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (lowerStr == "true" || lowerStr == "1") {
        return true;
    } else if (lowerStr == "false" || lowerStr == "0") {
        return false;
    } else {
        std::cerr << "Invalid boolean value: " << str << std::endl;
        return false; // 或者根据你的需要处理无效输入
    }
}


int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cout << "Usage: ./read_pic path_to_local_pic debug_flag !" << std::endl;
        return 0;
    }
    XReader xReader;
    std::string picture_dir = argv[1];
    std::string debug_string = argv[2];
    bool debug_flag = stringToBool(debug_string);
    if (debug_flag)
    {
        xReader.ShowLocalPicture(picture_dir);
        xReader.ExtractORBFeatures(picture_dir);
        xReader.ExtractFigureEdge(picture_dir);
        xReader.FigureSharpening(picture_dir);
        xReader.FigureGaussianBlurring(picture_dir);
        // xReader.FigureZoomInAndOut(picture_dir);
        xReader.AdaptiveFilterFigure(picture_dir);
    }
}