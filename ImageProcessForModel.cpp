#include "ImageProcessForModel.h"

static void resizeByScale(const cv::Size& modelSize, const cv::Mat& source, cv::Mat& destination)
{
    if (modelSize == source.size())
    {
        destination = source.clone();
        return;
    }

    double yScaleFactor = modelSize.height / static_cast<double>(source.rows);
    double xScaleFactor = modelSize.width / static_cast<double>(source.cols);
    //TODO: Determine when the image should be rotated
        cv::resize(source, destination, cv::Size{}, xScaleFactor, yScaleFactor, cv::INTER_LANCZOS4);
}
static void predictAndAnnotate(DarkHelp::NN& nn, const cv::Mat& source, cv::Mat& annotatedImage)
{
    nn.predict(source);
    annotatedImage = nn.annotate();
}

static void rotateImage(const cv::Mat& source, cv::Mat& destination, const double angle, const double scale)
{
    cv::Point2f center {source.cols / 2.0f, source.rows / 2.0f};
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, scale);
    cv::warpAffine(source, destination, rotation_matrix, source.size());
}