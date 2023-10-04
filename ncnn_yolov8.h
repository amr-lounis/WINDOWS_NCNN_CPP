#ifndef YOLOV8_H
#define YOLOV8_H

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <net.h>
#include "cpu.h"
#include <vector>
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
struct Object
{
    cv::Rect_<float> box;
    int class_id;
    float confidence;
};
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class FpsCalculator {
public:
    void update() {
        if (indix == 0) {
            start_time = std::chrono::steady_clock::now();
        }
        if (indix < max_frame) {
            indix++;
        }
        else {
            indix = 0;
            auto end_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            double t = duration.count() / 1000000.0;

            avg_temp = t / max_frame;
            avg_fps = 1 / avg_temp;
            if (max_fps < avg_fps) max_fps = avg_fps;
            printf_s("------------------------------\n");
            printf_s("avg_temp %f \n", avg_temp);
            printf_s("avg_fps %.2f \n", avg_fps);
            printf_s("max_fps %.2f \n", max_fps);
        }
    }
    double get_avg_temp() {
        return this->avg_temp;
    }
    double get_avg_fps() {
        return this->avg_fps;
    }
    double get_max_fps() {
        return this->max_fps;
    }

private:
    int indix = 0;
    const double max_frame = 30;
    double avg_temp = 0;
    double avg_fps = 0;
    double max_fps = 0;
    std::chrono::steady_clock::time_point start_time;
};
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ncnn_YoloV8
{
public:
    void init(const char* path_param, const  char* path_bin, int imgsz, bool gpu);
    void detect(const cv::Mat& rgb, std::vector<Object>& objects, float threshold_confidence, float threshold_nms);
private:
    void NMSBoxes_custom(const std::vector<cv::Rect>& _boxes, std::vector<int>& _indexes_nms, float _nms_threshold);
    float imageProcess(const cv::Mat& image_in, ncnn::Mat& image_out, int imgsz);
    float getValueAtPosition(const ncnn::Mat& mat, int col, int row);
    void generate_proposals(const ncnn::Mat& mat, float scale, float threshold_confidence, std::vector<cv::Rect>& boxes, std::vector<float>& confidences, std::vector<int>& class_ids);

    ncnn::Net model;
    int imgsz;
    bool gpu = false;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class OCR
{
public:
    void init( const char* path_param, const  char* path_bin, int imgsz, bool gpu);
    void read(const cv::Mat& rgb, std::string& text);
    ~OCR() {
        delete ocr;
        ocr = 0;
    }
private:
    //-------------------------------------------------------------------------------------------
    std::vector<std::string> array_names_ocr = {
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
            "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
            "U", "V", "W", "X", "Y", "Z"
    };
    ncnn_YoloV8* ocr;
};
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class GARIAMAN {
public:
    void init(bool use_gpu);
    void update(cv::Mat& image);
    ~GARIAMAN() {
        delete p_detector;
        p_detector = 0;

        delete p_ocr;
        p_ocr = 0;
    }
private:
    FpsCalculator fpsCalculator;
    ncnn_YoloV8* p_detector;
    OCR* p_ocr;
    //--------------------------
    bool isCroped(int w, int h, cv::Rect cropRegion);
    void drawRect(cv::Mat& rgb, const std::vector<Object>& objects);
    void writeFPS(cv::Mat& rgb, float avg_fps);
    void writeText(cv::Mat& rgb, std::string text, int x, int y);
};
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#endif // YOLOV8_H
