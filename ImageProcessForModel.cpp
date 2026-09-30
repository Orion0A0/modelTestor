#include "ImageProcessForModel.h"

#define ROTATE_ANGLE -90.0
#define ROTATE_SCALE 1
#define FLOAT_TWO 2.0f

//#TODO image distortion still exist, check out croping black border
void ImageProcessForModel::resizeByScale(const cv::Size& modelSize, const cv::Mat& source, cv::Mat& destination)
{
    if (modelSize == source.size())
    {
        destination = source.clone();
        return;
    }
    double xScaleFactor = modelSize.width / static_cast<double>(source.cols);
    double yScaleFactor = modelSize.height / static_cast<double>(source.rows);
    Dimension modelDimension = findDimension(modelSize);
    Dimension imageDimension = findDimension(source.size());

    if (modelDimension == RECTANGLE_LONGER_HEIGHT && imageDimension == RECTANGLE_LONGER_WIDTH ||
        modelDimension == RECTANGLE_LONGER_WIDTH && imageDimension == RECTANGLE_LONGER_HEIGHT)
    {
        cv::Mat temp;
        rotateImage(source, temp, ROTATE_ANGLE, ROTATE_SCALE);
        cv::resize(temp, destination, cv::Size{}, xScaleFactor, yScaleFactor, cv::INTER_LANCZOS4);
        return;
    }
    // else if modelDimension is SQUARE
        cv::resize(source, destination, cv::Size{}, xScaleFactor, yScaleFactor, cv::INTER_LANCZOS4);
}
void ImageProcessForModel::predictAndAnnotate(DarkHelp::NN& nn, const cv::Mat& source, cv::Mat& annotatedImage)
{
    nn.predict(source);
    annotatedImage = nn.annotate();
}

void ImageProcessForModel::rotateImage(const cv::Mat& source, cv::Mat& destination, const double angle, const double scale)
{
    cv::Point2f center {source.cols / FLOAT_TWO, source.rows / FLOAT_TWO};
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, scale);
    cv::warpAffine(source, destination, rotation_matrix, source.size());
}

Dimension ImageProcessForModel::findDimension(const cv::Size& modelSize)
{
    if (modelSize.width > modelSize.height)
        return RECTANGLE_LONGER_WIDTH;
    if (modelSize.height > modelSize.width)
        return RECTANGLE_LONGER_HEIGHT;

    return SQUARE;
}
