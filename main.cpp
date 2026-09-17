#include <filesystem>

#include <DarkHelp.hpp>
#include <opencv2/imgcodecs.hpp>

#define POKER_FOLDER "/poker_model/"
#define CONFIG_FILE "poker.cfg"
#define WEIGHTS_FILE "poker_best.weights"
#define NAMES_FILE "poker.names"
#define TEST_IMG_PARENT_FOLDER "/testImage"
#define TEST_IMG_SRC_FOLDER "/src"
#define TEST_IMG_DST_FOLDER "/dst"

static void resizeByScale(const cv::Size& modelSize, const cv::Mat& source, cv::Mat& destination);
static void predictAndAnnotate(DarkHelp::NN& nn, const cv::Mat& source, cv::Mat& annotatedImage);
static void rotateImage(const cv::Mat& source, cv::Mat& destination, int angle);
int main(int argc, char** argv)
{
    // built inside build directory, so need to go back one step
    std::string currDir = std::filesystem::current_path().parent_path().string();

    DarkHelp::Config cfg (currDir + POKER_FOLDER + CONFIG_FILE, currDir + POKER_FOLDER +  WEIGHTS_FILE, currDir + POKER_FOLDER + NAMES_FILE);
    cfg.names_include_percentage = true;
    cfg.annotation_auto_hide_labels = false;
    cfg.enable_tiles = false;
    cfg.combine_tile_predictions = true;
    DarkHelp::NN nn (cfg);

    std::filesystem::path src {currDir + TEST_IMG_PARENT_FOLDER + TEST_IMG_SRC_FOLDER};
    std::filesystem::path dst {currDir + TEST_IMG_PARENT_FOLDER + TEST_IMG_DST_FOLDER};

    cv::Mat originImage;
    cv::Mat resizedImage;
    cv::Mat annotatedImage;
    int fileNumber = 1;
    for (const auto& entry: std::filesystem::directory_iterator{src})
    {
        if (!entry.exists())
        {
            std::cout << entry.path() << " does not exist" << std::endl;
            continue;
        }
        originImage = cv::imread(entry.path());

        // check if image is read successfully
        if (src.empty())
            continue;

        resizeByScale(nn.network_size(), originImage, resizedImage);
        predictAndAnnotate(nn, resizedImage, annotatedImage);

        cv::imwrite(dst.string() + "/" + std::to_string(fileNumber++) + ".jpg", annotatedImage);
    }

}
void resizeByScale(const cv::Size& modelSize, const cv::Mat& source, cv::Mat& destination)
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
void predictAndAnnotate(DarkHelp::NN& nn, const cv::Mat& source, cv::Mat& annotatedImage)
{
    nn.predict(source);
    annotatedImage = nn.annotate();
}
//TODO: write the code the rotate the image
static void rotateImage(const cv::Mat& source, cv::Mat& destination, int angle)
{

}