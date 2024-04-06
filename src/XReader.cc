#include "XReader.h"
#include <fstream>
#include <vector>

namespace
{
    const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};

    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    const float SCORE_THRESHOLD = 0.2;
    const float NMS_THRESHOLD = 0.4;
    const float CONFIDENCE_THRESHOLD = 0.4;

    cv::Mat format_yolov5(const cv::Mat &source)
    {
        const int col = source.cols;
        const int row = source.rows;
        const int _max = MAX(col, row);
        cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
        source.copyTo(result(cv::Rect(0, 0, col, row)));
        return result;
    }

    void VisualizeOutputResults(const std::vector<XReader::DetectionClass> &out_put_class, const std::vector<std::string> &class_list, cv::Mat *image)
    {
        const size_t detections = out_put_class.size();
        for (size_t index = 0; index < detections; ++index)
        {
            auto detection = out_put_class[index];
            auto box = detection.box;
            auto classId = detection.class_id;
            const auto color = colors[classId % colors.size()];
            cv::rectangle(*image, box, color, 3);
            cv::rectangle(*image, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
            cv::putText(*image, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
        }
        cv::imshow("Detected Class", *image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    cv::Mat GaussianBlurProcess(const cv::Mat &image)
    {
        cv::Mat gray_image;
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
        cv::Mat blurred;
        cv::GaussianBlur(gray_image, blurred, cv::Size(5, 5), 0);
        return blurred;
    }
}

std::vector<std::string> XReader::LoadClassList()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("/home/jaysszhou/Documents/Algorithm/functions/Peception/vision/lib/classes.txt");
    std::string line;
    while (getline(ifs, line))
    {
        class_list.emplace_back(line);
    }
    return class_list;
}

void XReader::LoadNeuralNetwork(cv::dnn::Net *net, bool is_cuda)
{
    cv::dnn::Net result = cv::dnn::readNet("/home/jaysszhou/Documents/Algorithm/functions/Peception/vision/lib/yolov5s.onnx");
    if (result.empty())
    {
        std::cerr << "Failed to load network." << std::endl;
    }
    else
    {
        std::cout << "Network loaded successfully." << std::endl;
    }
    if (is_cuda)
    {
        std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    *net = result;
}

void XReader::RecognizePictureClassByYoLo5(const cv::Mat &image, cv::dnn::Net &net, const std::vector<std::string> &className, std::vector<DetectionClass> &output)
{
    if (image.empty() || className.empty())
    {
        std::cout << "[XREADER] check image or class_list !! " << std::endl;
        return;
    }
    cv::Mat blob;
    std::vector<cv::Mat> outputs;
    auto input_image = format_yolov5(image);
    try
    {
        cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
        net.setInput(blob);
        net.forward(outputs, net.getUnconnectedOutLayersNames());
    }
    catch (const cv::Exception &e)
    {
        std::cerr << "Error during forward pass: " << e.what() << std::endl;
    }

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float *data = (float *)outputs[0].data;
    const int rows = 25200;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < rows; ++i)
    {

        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD)
        {

            float *classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD)
            {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }

        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (size_t i = 0; i < nms_result.size(); i++)
    {
        int idx = nms_result[i];
        DetectionClass result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}

void XReader::ShowLocalPicture(const std::string &local_pic_dir)
{
    cv::Mat image = cv::imread(local_pic_dir, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cout << "[XReader] Load picture from " << local_pic_dir << " failed " << std::endl;
        return;
    }
    cv::imshow("Loaded Image", image);
    cv::waitKey(0);
    bool is_cuda = false;
    cv::dnn::Net yolov5;
    std::vector<std::string> class_list = LoadClassList();
    LoadNeuralNetwork(&yolov5, is_cuda);
    std::vector<DetectionClass> out_put_class;
    RecognizePictureClassByYoLo5(image, yolov5, class_list, out_put_class);
    VisualizeOutputResults(out_put_class, class_list, &image);
}

void XReader::ExtractORBFeatures(const std::string &local_pic_dir)
{
    cv::Mat image = cv::imread(local_pic_dir, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cout << "[XReader] Load picture from " << local_pic_dir << " failed " << std::endl;
        return;
    }
    cv::Mat grayImg;
    cv::cvtColor(image, grayImg, cv::COLOR_BGR2GRAY);

    int nfeatures = 2000;     // 最大特征点数量
    float scaleFactor = 1.2f; // 金字塔缩放因子
    int nlevels = 8;          // 金字塔层数
    cv::Ptr<cv::ORB> orb = cv::ORB::create(nfeatures, scaleFactor, nlevels);
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
    cv::Mat img_keypoints;
    cv::drawKeypoints(image, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB KeyPoints", img_keypoints);
    cv::waitKey(0);
}

void XReader::DetectCannyEdge(const cv::Mat &image)
{
    std::cout << "[XReader] Show canny edge !" << std::endl;
    cv::Mat edges;
    cv::Canny(image, edges, 100, 200);
    cv::imshow("Canny Edge Detection", edges);
    cv::waitKey(0);
}

void XReader::DetectLaplacianEdge(const cv::Mat &image)
{
    std::cout << "[XReader] Show Laplacian edge !" << std::endl;
    cv::Mat edges;
    cv::Laplacian(image, edges, CV_16S, 3);
    cv::convertScaleAbs(edges, edges);
    cv::imshow("Laplacian Edge Detection", edges);
    cv::waitKey(0);
}

void XReader::DetectSobelEdge(const cv::Mat &image)
{
    std::cout << "[XReader] Show Sobel edge !" << std::endl;
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y, grad;
    // 计算x和y方向上的梯度
    Sobel(image, grad_x, CV_16S, 1, 0, 3);
    convertScaleAbs(grad_x, abs_grad_x);
    Sobel(image, grad_y, CV_16S, 0, 1, 3);
    convertScaleAbs(grad_y, abs_grad_y);

    // 合并梯度
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    cv::imshow("Sobel Edge Detection", grad);
    cv::waitKey(0);
}

void XReader::ExtractFigureEdge(const std::string &local_pic_dir)
{
    const cv::Mat image = cv::imread(local_pic_dir, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cout << "[XReader] Load picture from " << local_pic_dir << " failed " << std::endl;
        return;
    }
    DetectCannyEdge(image);
    DetectLaplacianEdge(image);
    DetectSobelEdge(image);
}