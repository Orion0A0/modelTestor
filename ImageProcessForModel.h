/*
 * Created by Orion on 9/17/26.
 *
 * The class helps to do image processing for model trained from DarkNet
 */

#include "DarkHelpNN.hpp"


#ifndef POKER_IMAGEPROCESSING_H
#define POKER_IMAGEPROCESSING_H

enum Dimension {SQUARE, RECTANGLE_LONGER_HEIGHT, RECTANGLE_LONGER_WIDTH};

class ImageProcessForModel
{
public:
    static void resizeByScale(const cv::Size& modelSize, const cv::Mat& source, cv::Mat& destination);
    static void predictAndAnnotate(DarkHelp::NN& nn, const cv::Mat& source, cv::Mat& annotatedImage);
    static void rotateImage(const cv::Mat& source, cv::Mat& destination, double angle, double scale);
private:
    static Dimension findDimension(const cv::Size& modelSize);
};


#endif //POKER_IMAGEPROCESSING_H
