#ifndef XREADER_H
#define XREADER_H

#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>

class XReader
{
public:
    XReader()
    {
        std::cout << " Build XReader" << std::endl;
    };
    ~XReader()
    {
        std::cout << " Delete XReader" << std::endl;
    };

    struct DetectionClass
    {
        int class_id;
        float confidence;
        cv::Rect box;
    };
    void ShowLocalPicture(const std::string &local_pic_dir);

private:
    std::vector<std::string> LoadClassList();
    void LoadNeuralNetwork(cv::dnn::Net *net, bool is_cuda);
    void RecognizePictureClassByYoLo5(const cv::Mat &image, cv::dnn::Net &net, const std::vector<std::string> &className, std::vector<DetectionClass> &output);
};

#endif