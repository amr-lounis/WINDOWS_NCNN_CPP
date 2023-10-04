#include "ncnn_yolov8.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <filesystem>

using namespace std;

int main(int argc, char** argv)
{
    GARIAMAN* ga = new  GARIAMAN();
    ga->init(false);
    cv::Mat imageG = cv::imread("image0.jpg");

    int itirate = 300;
    for (int i = 0; i <= itirate; i++) {
        cv::Mat image = imageG.clone();
        ga->update(image);

        if (i >= itirate) {
            cv::imwrite("test.jpg", image);
        }
    }
    return 0;
}