#include<iostream>
#include<vector>
#include<memory>
#include<string>
#include "torch/torch.h"
#include "torch/script.h"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class YOLOV3SPP {
 public:
	 YOLOV3SPP() { }
	 void Initialize(torch::Device *device);
	 void Deinitialize();
	 std::vector<std::vector<int>> Forward(cv::Mat& image);
	 ~YOLOV3SPP() = default;
 private:
	 void SetFixedParams();
	 void LoadTracedModule();
	 std::vector<std::vector<int>> DetectionLayer(torch::Tensor pred, std::vector<float>& cls_scores);
	 void Decoder(torch::Tensor& box);
	 torch::Tensor nms(const torch::Tensor& decode_loc, const torch::Tensor& conf);
	 cv::Mat Letterbox(cv::Mat src_img, int new_size);
	 void scale_loc(std::vector<std::vector<int>>& result, cv::Mat input, cv::Mat origin_pic);
	 std::shared_ptr<torch::jit::script::Module> model_;
	 std::vector<int> image_size_;
	 std::string weight_path_;
	 torch::Device *device_;
	 float nms_threshold_;
	 float conf_threshold_;
	 std::vector <std::string> classnames_;
};