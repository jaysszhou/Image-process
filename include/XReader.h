#ifndef XREADER_H
#define XREADER_H

#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/dnn/dnn.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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

    enum adaptiveMethod{meanFilter,gaussianFilter,medianFilter};
    void ShowLocalPicture(const std::string &local_pic_dir);
    void ExtractORBFeatures(const std::string &local_pic_dir);
    void ExtractFigureEdge(const std::string &local_pic_dir);
    void FigureSharpening(const std::string &local_pic_dir);
    void FigureGaussianBlurring(const std::string &local_pic_dir);
    void FigureZoomInAndOut(const std::string&local_pic_dir);
    void AdaptiveFilterFigure(const std::string& local_pic_dir);
private:
    std::vector<std::string> LoadClassList();
    void LoadNeuralNetwork(cv::dnn::Net *net, bool is_cuda);
    void RecognizePictureClassByYoLo5(const cv::Mat &image, cv::dnn::Net &net, const std::vector<std::string> &className, std::vector<DetectionClass> &output);
    void DetectCannyEdge(const cv::Mat&image);
    void DetectLaplacianEdge(const cv::Mat&image);
    void DetectSobelEdge(const cv::Mat&image);
    void AdaptiveThreshold(cv::Mat& src, cv::Mat& dst, double Maxval, int Subsize, double c, adaptiveMethod method);
};

#endif