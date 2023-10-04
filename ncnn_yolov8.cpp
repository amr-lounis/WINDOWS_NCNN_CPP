#include "ncnn_yolov8.h"
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void ncnn_YoloV8::init(const char* path_param, const  char* path_bin, int imgsz, bool gpu)
{
    printf_s("init model -------------------------------- \n");
    printf_s("path param :%s \n", path_param);
    printf_s("path bin   :%s \n", path_bin);
    printf_s("imgsz      :%d \n", imgsz);

    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);

    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    this->imgsz = imgsz;


    model.clear();

    model.opt = ncnn::Option();
#if NCNN_VULKAN
    model.opt.use_vulkan_compute = gpu;
#endif
    model.opt.num_threads = ncnn::get_big_cpu_count();
    model.opt.blob_allocator = &blob_pool_allocator;
    model.opt.workspace_allocator = &workspace_pool_allocator;
    //
    model.load_param( path_param);
    model.load_model( path_bin);
}
void ncnn_YoloV8::detect(const cv::Mat& rgb, std::vector<Object>& objects, float threshold_confidence, float threshold_nms)
{
    objects.clear();
    ncnn::Mat mat_in;
    ncnn::Mat mat_out;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<int> indexes_nms;

    float scale = imageProcess(rgb, mat_in, imgsz);

    ncnn::Extractor ex = model.create_extractor();
    ex.input(model.input_names()[0], mat_in);
    ex.extract(model.output_names()[0], mat_out);

    generate_proposals(mat_out, scale, threshold_confidence, boxes, confidences, class_ids);

    NMSBoxes_custom(boxes, indexes_nms, threshold_nms);

    for (int i : indexes_nms) {
        Object obj;
        obj.box = boxes[i];
        obj.class_id = class_ids[i];
        obj.confidence = confidences[i];
        objects.push_back(obj);
    }
}
void ncnn_YoloV8::NMSBoxes_custom(const std::vector<cv::Rect>& _boxes, std::vector<int>& _indexes_nms, float _nms_threshold)
{
    _indexes_nms.clear();
    const int n = _boxes.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)  areas[i] = _boxes[i].width * _boxes[i].height;
    for (int i = 0; i < n; i++)
    {
        const cv::Rect& a = _boxes[i];

        int keep = 1;
        for (int j = 0; j < (int)_indexes_nms.size(); j++)
        {
            const cv::Rect& b = _boxes[_indexes_nms[j]];
            // intersection over union
            float inter_area = (a & b).area();
            float union_area = areas[i] + areas[_indexes_nms[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > _nms_threshold)
                keep = 0;
        }

        if (keep)
            _indexes_nms.push_back(i);
    }
}
float ncnn_YoloV8::imageProcess(const cv::Mat& image_in, ncnn::Mat& image_out, int imgsz) {
    int height = image_in.rows;
    int width = image_in.cols;
    int length = (std::max)(height, width);

    cv::Mat image_nn3(length, length, CV_8UC3, cv::Scalar(0, 0, 0));
    image_in.copyTo(image_nn3(cv::Rect(0, 0, width, height)));
    float scale = (float)length / (float)imgsz;

    image_out = ncnn::Mat::from_pixels_resize(image_nn3.data, ncnn::Mat::PIXEL_RGB2BGR, length, length, imgsz, imgsz);

    float norm_vals[] = { 1.0 / 255.0f,1.0 / 255.0f ,1.0 / 255.0f };
    image_out.substract_mean_normalize(0, norm_vals);

    return scale;
}
float ncnn_YoloV8::getValueAtPosition(const ncnn::Mat& mat, int col, int row) {
    return mat[row * mat.w + col];
}
void ncnn_YoloV8::generate_proposals(const ncnn::Mat& mat, float scale, float threshold_confidence, std::vector<cv::Rect>& boxes, std::vector<float>& confidences, std::vector<int>& class_ids) {
    boxes.clear();
    confidences.clear();
    class_ids.clear();

    for (int c = 0; c < mat.w; c++) {
        float max = 0;
        int class_id = 0;
        for (int i = 4; i < mat.h; i++) {
            float value = getValueAtPosition(mat, c, i);
            if (value > max) {
                max = value;
                class_id = i - 4;
            }
        }
        if (max > threshold_confidence) {
            int cx = getValueAtPosition(mat, c, 0);
            int cy = getValueAtPosition(mat, c, 1);
            int w = getValueAtPosition(mat, c, 2);
            int h = getValueAtPosition(mat, c, 3);

            int left = int((cx - (0.5 * w)) * scale);
            int top = int((cy - (0.5 * h)) * scale);
            int width = int(w * scale);
            int height = int(h * scale);

            boxes.emplace_back(cv::Rect(left, top, width, height));
            confidences.push_back(max);
            class_ids.push_back(class_id);
        }
    }
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void OCR::init(const char* path_param, const  char* path_bin, int imgsz, bool gpu) {
    ocr = new ncnn_YoloV8();
    ocr->init( path_param, path_bin, imgsz, gpu);
}
void OCR::read(const cv::Mat& rgb, std::string& text) {
    std::vector<Object> objects;
    ocr->detect(rgb, objects, 0.4f, 0.2f);
    std::sort(objects.begin(), objects.end(), [](const Object& a, const Object& b) {
        return a.box.x < b.box.x;
        });
    std::vector<char> list_char;
    for (const Object& o_ocr : objects) {
        int id_object = o_ocr.class_id;
        char character = array_names_ocr[id_object][0];
        list_char.push_back(character);
    }
    std::string plate_txt(list_char.begin(), list_char.end());
    text = plate_txt;
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void GARIAMAN::init( bool use_gpu) {
    if (this->p_detector != 0) {
        delete this->p_detector;
        this->p_detector = 0;
    }
    if (this->p_ocr != 0) {
        delete this->p_ocr;
        this->p_ocr = 0;
    }

    if (this->p_detector == 0) {
        this->p_detector = new ncnn_YoloV8();
        this->p_detector->init("_car+plate_320.param", "_car+plate_320.bin", 320, use_gpu);
    }

    if (this->p_ocr == 0) {
        this->p_ocr = new OCR();
        this->p_ocr->init("_ocr_224.param", "_ocr_224.bin", 224, use_gpu);
    }
}
void GARIAMAN::update(cv::Mat& image) {
    if (p_detector != 0 || p_ocr != 0) {
        std::vector<Object> objects;
        p_detector->detect(image, objects, 0.4f, 0.2f);
        for (Object o : objects) {
            if ((o.class_id == 1) && isCroped(image.cols, image.rows, o.box)) {
                cv::Mat cropped_image = image(o.box);
                std::string text = "";
                p_ocr->read(cropped_image, text);
                writeText(image, text, o.box.x, o.box.y - 20);
            }
            if (o.class_id == 0)
            {
                char c[10];
                sprintf_s(c, "CAR:%.2d%%", (int)(o.confidence * 100));
                std::string text = c;
                writeText(image, text, o.box.x, o.box.y);
            }
        }
        drawRect(image, objects);
    }
    else {
        writeText(image, "ERROR:not init model", 0, 0);
    }
    //--------------------------------------------------
    fpsCalculator.update();
    writeFPS(image, fpsCalculator.get_avg_fps());
}
bool GARIAMAN::isCroped(int w, int h, cv::Rect cropRegion) {
    if (cropRegion.x >= 0 && cropRegion.y >= 0 &&
        cropRegion.width > 0 && cropRegion.height > 0 &&
        cropRegion.x + cropRegion.width <= w &&
        cropRegion.y + cropRegion.height <= h) {
        return true;
    }
    else {
        return false;
    }
}
void GARIAMAN::drawRect(cv::Mat& rgb, const std::vector<Object>& objects)
{
    for (size_t i = 0; i < objects.size(); i++)
    {
        cv::rectangle(rgb, objects[i].box, cv::Scalar(255, 0, 0));
    }
}
void GARIAMAN::writeFPS(cv::Mat& rgb, float avg_fps) {
    char text[32];
    sprintf_s(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
        cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

}
void GARIAMAN::writeText(cv::Mat& rgb, std::string text, int x, int y) {
    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
        cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
}
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++