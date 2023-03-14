//
// Created by MQM on 2023.03.13.
//
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

void fill_tensor_data_image(ov::Tensor& input_tensor, const cv::Mat& input_image) {
    // 获取输入节点要求的输入图片数据的大小
    ov::Shape tensor_shape = input_tensor.get_shape();
    const size_t width = tensor_shape[3]; // 要求输入图片数据的宽度
    const size_t height = tensor_shape[2]; // 要求输入图片数据的高度
    const size_t channels = tensor_shape[1]; // 要求输入图片数据的维度
    // 读取节点数据内存指针
    float* input_tensor_data = input_tensor.data<float>();
    // 将图片数据填充到网络中
    // 原有图片数据为 H、W、C 格式，输入要求的为 C、H、W 格式
    for (size_t c = 0; c < channels; c++) {
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                input_tensor_data[c * width * height + h * width + w] = input_image.at<cv::Vec<float, 3>>(h, w)[c];
            }
        }
    }
}
int main(){
    std::vector<std::string> class_names = { "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light","fire hydrant",
                                   "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe","backpack", "umbrella",
                                   "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove","skateboard", "surfboard",
                                   "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange","broccoli", "carrot",
                                   "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse","remote",
                                   "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
    cv::Mat image = cv::imread("/Volumes/TNIT/Datasets/SanLi/Test/Snipaste_2023-03-11_17-06-39.png");

    ov::Core core;
    std::string onnxPath = "/Users/mqm/WORKSPACE/OPENVINO/models/work/SanLi/DianHan.onnx";
    std::string xmlPath ="/Users/mqm/WORKSPACE/OPENVINO/models/work/SanLi/DianHan.xml";
    ov::CompiledModel compiled_model = core.compile_model(xmlPath, "AUTO");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    int64 start = cv::getTickCount();

    // Get input port for model with one input
    auto input_port = compiled_model.input();

    int input_height = input_port.get_shape()[2];
    int input_width =  input_port.get_shape()[3];

    float x_factor = image.cols / input_width;
    float y_factor = image.rows / input_height;

    std::cout << "input_h:" << input_height << "; input_w:" << input_width << std::endl;

    cv::Mat blob_image;
    resize(image, blob_image, cv::Size(input_width, input_height));
    blob_image.convertTo(blob_image, CV_32F);
    blob_image = blob_image / 255.0;

    // Create tensor from external memory
    float* input_data = (float*)blob_image.data;

    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), input_data);
    // Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor);

    infer_request.start_async();
    infer_request.wait();

    // Get output tensor by tensor name
    ov::Tensor output_tensor = infer_request.get_tensor("output");
    int out_rows = output_tensor.get_shape()[1]; //获得"output"节点的out_rows
    int out_cols = output_tensor.get_shape()[2]; //获得"output"节点的Width
    std::cout << "output_h:" << input_height << "; output_w:" << input_width << std::endl;


    const float* output_buffer = output_tensor.data<const float>();
// output_buffer[] - accessing output tensor data
    if(output_buffer){
        cv::Mat det_output(out_rows, out_cols, CV_32F, (float*)output_buffer);
        std::vector<cv::Rect> boxes;
        std::vector<int> classIds;
        std::vector<float> confidences;

        for (int i = 0; i < det_output.rows; i++) {
            float confidence = det_output.at<float>(i, 4);
            if (confidence < 0.2 || std::isnan(confidence) ) {
                continue;
            }
            cv::Mat classes_scores = det_output.row(i).colRange(5, 7);
            cv::Point classIdPoint;
            double score;
            minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

            // 置信度 0～1之间
            if (score > 0.5)
            {
                float cx = det_output.at<float>(i, 0);
                float cy = det_output.at<float>(i, 1);
                float ow = det_output.at<float>(i, 2);
                float oh = det_output.at<float>(i, 3);
                int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
                int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
                int width = static_cast<int>(ow * x_factor);
                int height = static_cast<int>(oh * y_factor);
                cv::Rect box;
                box.x = x;
                box.y = y;
                box.width = width;
                box.height = height;

                boxes.push_back(box);
                classIds.push_back(classIdPoint.x);
                confidences.push_back(score);
            }
        }
        // NMS
        std::vector<int> indexes;
        cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.45, indexes);
        for (size_t i = 0; i < indexes.size(); i++) {
            int index = indexes[i];
            int idx = classIds[index];
            cv::rectangle(image, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
            cv::rectangle(image, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
                          cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
            cv::putText(image, class_names[idx], cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
        }

        // 计算FPS render it
        float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
        std::cout << "Infer time(ms): " << t * 1000 << "ms; Detections: " << indexes.size() << std::endl;
        putText(image, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
        cv::imshow("YOLOv5-6.1 + OpenVINO 2022.1 C++ Demo", image);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
    return 1;
}