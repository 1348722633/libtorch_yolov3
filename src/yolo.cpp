#include "yolo.hpp"
#include <math.h>
#include <iomanip>
using namespace cv;
using namespace std;
void YOLOV3SPP::SetFixedParams() {
	image_size_ = {320, 512};
	nms_threshold_ = 0.6;
	conf_threshold_ = 0.3;
	classnames_ = { "person", "bicycle", "car", "motorcycle", "airplane", "bus",
		"train", "truck", "boat", "traffic light", "fire hydrant",
		"stop sign", "parking meter", "bench", "bird", "cat", "dog",
		"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
		"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat",
		"baseball glove", "skateboard", "surfboard", "tennis racket",
		"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
		"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
		"hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop",
		"mouse", "remote", "keyboard", "cell phone", "microwave",
		"oven", "toaster", "sink", "refrigerator", "book", "clock",
		"vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
}

void YOLOV3SPP::LoadTracedModule() {
	weight_path_ = "D:/code/libtorch_yolov3/model/yolomodel.pt";
	model_ = std::make_shared<torch::jit::script::Module>(torch::jit::load(weight_path_));
	model_->to(*device_);
	model_->eval();
}

void YOLOV3SPP::Initialize(torch::Device *device) {
	SetFixedParams();
	device_ = device;
	LoadTracedModule();
}

cv::Mat YOLOV3SPP::Letterbox(cv::Mat src_image, int new_size) {
	int height = src_image.rows; int width = src_image.cols;
	vector<int> new_shape = { new_size, new_size };
	float r = min(float(new_shape[0])/height, float(new_shape[1])/width);
	vector<int> new_unpad = { int(round(width * r)), int(round(height * r)) };
	int dw = new_shape[1] - new_unpad[0]; int dh = new_shape[0] - new_unpad[1];
	//对64取模
	dw = dw % 64; dh = dh % 64;
	dw = dw / 2; dh = dh / 2;
	cv::Mat dst_img = src_image;
	cv::resize(dst_img, dst_img, cv::Size(new_unpad[0], new_unpad[1]), INTER_LINEAR);
	int top = int(round(dh - 0.1)); int bottom = int(round(dh + 0.1));
	int left = int(round(dw - 0.1)); int right = int(round(dw - 0.1));
	cv::copyMakeBorder(dst_img, dst_img, top, bottom, left, right, BORDER_CONSTANT, cv::Scalar(114,114,114));
	return dst_img;
}

void YOLOV3SPP::Decoder(torch::Tensor& box) {
	box.slice(1, 0, 2) = box.slice(1, 0, 2) - box.slice(1, 2, 4).div(2);
	// 此时的框的坐标从中心点变成了左上顶点，所以右下顶点要加上框。
	box.slice(1, 2, 4) = box.slice(1, 0, 2) + box.slice(1, 2, 4);
}

torch::Tensor YOLOV3SPP::nms(const torch::Tensor& decode_loc, const torch::Tensor& conf) {
	torch::Tensor keep = torch::empty({ conf.sizes()[0] }).to(torch::kLong).to(conf.device());
	torch::Tensor x1, y1, x2, y2;
	x1 = decode_loc.select(1, 0); y1 = decode_loc.select(1, 1);
	x2 = decode_loc.select(1, 2);y2 = decode_loc.select(1, 3);
	torch::Tensor area = torch::mul(x2 - x1, y2 - y1); 
	std::tuple<torch::Tensor, torch::Tensor> sort_result = torch::sort(conf, 0, true); // 分数从高到低排列
	torch::Tensor scores = std::get<0>(sort_result).squeeze(1); 
	torch::Tensor sorted_idx = std::get<1>(sort_result).squeeze(1);
	int count = 0;
	while (sorted_idx.numel()) {
		auto i = sorted_idx[0];
		keep[count] = i;
		count++;
		if (sorted_idx.sizes()[0] == 1) break;
		sorted_idx = torch::slice(sorted_idx, 0, 1, sorted_idx.size(0));
		torch::Tensor xx1 = torch::index_select(x1, 0, sorted_idx);
		torch::Tensor xx2 = torch::index_select(x2, 0, sorted_idx);
		torch::Tensor yy1 = torch::index_select(y1, 0, sorted_idx);
		torch::Tensor yy2 = torch::index_select(y2, 0, sorted_idx);

		torch::Tensor inter_rect_x1 = torch::max(xx1, x1[i]);
		torch::Tensor inter_rect_y1 = torch::max(yy1, y1[i]);
		torch::Tensor inter_rect_x2 = torch::min(xx2, x2[i]);
		torch::Tensor inter_rect_y2 = torch::min(yy2, y2[i]);
		torch::Tensor inter_area = torch::max(inter_rect_x2 - inter_rect_x1,
			                       torch::zeros(inter_rect_x1.sizes()).to(inter_rect_x1.device())) *
			                       torch::max(inter_rect_y2 - inter_rect_y1,
				                   torch::zeros(inter_rect_y1.sizes()).to(inter_rect_y1.device()));
		torch::Tensor union_area = area.index_select(0, sorted_idx) - inter_area + area[i];
		inter_area = inter_area.div_(union_area);
		torch::Tensor inter_nms = (inter_area < nms_threshold_);
		auto mask_inx = torch::nonzero(inter_nms).squeeze(1);
		sorted_idx = torch::index_select(sorted_idx, 0, mask_inx);
	}
	return keep.slice(0, 0, count);
}




std::vector<std::vector<int>> YOLOV3SPP::DetectionLayer(torch::Tensor pred, std::vector<float>& cls_scores) {
	vector<vector<int>> result;
	pred = pred[0];
	/*pred 元素排列(x1, y1, x2, y2, conf, cls)*/
	//cout << "pred size" << pred.sizes() << endl;
	int minwh = 2; int maxwh = 4096;
	// 保留conf分数大于阈值的预测值
	torch::Tensor conf_index = (pred.select(1, 4) > conf_threshold_);
	conf_index = torch::nonzero(conf_index).squeeze(1);
	pred = torch::index_select(pred, 0, conf_index);
	// 保留预测框宽高在[2, 4096]之间的预测项
	torch::Tensor box_index = ((pred.slice(1, 2, 4) > minwh) * (pred.slice(1, 2, 4) < maxwh)).all(1);
	box_index = torch::nonzero(box_index).squeeze(1);
	pred = torch::index_select(pred, 0, box_index);
	if (pred.sizes()[0] == 0) {
		//返回
		return result;
	}
	int channels = pred.sizes()[1];
	// class_conf = class_conf * obj_conf;
	cout << pred.slice(1, 5, channels).sizes() << endl;
	cout << pred.slice(1, 4, 5).sizes();
	pred.slice(1, 5, channels) = pred.slice(1, 5, channels) * pred.slice(1, 4, 5);
	Decoder(pred.slice(1, 0, 4));
	std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(pred.slice(1, 5, channels), 1);
	torch::Tensor max_conf = std::get<0>(max_classes).to(torch::kFloat32);
	torch::Tensor max_conf_index = std::get<1>(max_classes).to(torch::kFloat32);
	torch::Tensor filter_result = torch::cat({pred.slice(1, 0, 4), max_conf.unsqueeze(1), max_conf_index.unsqueeze(1)}, 1);
	torch::Tensor class_index = (filter_result.select(1, 4) > conf_threshold_);
	class_index = torch::nonzero(class_index).squeeze(1);
	filter_result = torch::index_select(filter_result, 0, class_index);
	if (filter_result.sizes()[0] == 0) {
	  //返回
		return result;
	}
	torch::Tensor decode_loc = filter_result.slice(1, 0, 4);
	torch::Tensor scores = filter_result.select(1, 4).unsqueeze(1);
	cout << scores << endl;
	torch::Tensor class_idx = filter_result.select(1, 5).unsqueeze(1);
	torch::Tensor keep = nms(decode_loc, scores);

	decode_loc = torch::index_select(decode_loc, 0, keep);
	scores = torch::index_select(scores, 0, keep);
	class_idx = torch::index_select(class_idx, 0, keep);
	torch::Tensor final_result = torch::cat({decode_loc, scores, class_idx}, 1);

	// store the information in vector
	final_result = final_result.cpu();
	auto final_result_data = final_result.accessor<float, 2>();
	int size = final_result.size(0);
	for (int i = 0; i < size; i++) {
		// we only need person, if you need other class, you can change
		if (final_result_data[i][5] == 0) {
			vector<int> single_box;
			single_box = { int(final_result_data[i][0]), int(final_result_data[i][1]),
				int(final_result_data[i][2]), int(final_result_data[i][3]), int(final_result_data[i][5])};
			cls_scores.push_back(final_result_data[i][4]);
			result.push_back(single_box);
		}
	}
	return result;
	//std::cout << keep.sizes() << endl;;
	//cout << filter_result << endl;
}

void YOLOV3SPP::scale_loc(std::vector<std::vector<int>>& result, cv::Mat input, cv::Mat origin_pic) {
	int input_height = input.rows; int input_width = input.cols;
	int origin_height = origin_pic.rows; int origin_width = origin_pic.cols;
	float ratio = float(max(input_height, input_width)) / max(origin_height, origin_width);
	int pad_x = (origin_width * ratio - input_width)/2;
	int pad_y = (origin_height * ratio - input_height)/2;
	for (int i = 0; i < result.size(); i++) {
		result[i][0] = (result[i][0] + pad_x) / ratio; result[i][2] = (result[i][2] + pad_x) / ratio;
		result[i][1] = (result[i][1] + pad_y) / ratio; result[i][3] = (result[i][3] + pad_y) / ratio;
		result[i][0] = min(max(result[i][0], 0), origin_width);
		result[i][1] = min(max(result[i][1], 0), origin_height);
		result[i][2] = min(max(result[i][2], 0), origin_width);
		result[i][3] = min(max(result[i][3], 0), origin_height);
	}
}

std::vector<std::vector<int>> YOLOV3SPP::Forward(cv::Mat& image) {
	std::vector<std::vector<int>> result;
	std::vector<float> cls_scores;
	cv::resize(image, image, cv::Size(1280, 720));
	cv::Mat input = image;
	cv::Mat pic = image; 
	//cv::resize(pic, pic,cv::Size(512, 320));
	input = Letterbox(input, image_size_[1]);
	cv::cvtColor(input, input, CV_BGR2RGB);
	input.convertTo(input, CV_32F, 1.0/255);
	int kImageHeight = input.rows; int kImageWidth = input.cols; int kChannels = input.channels();
	auto input_tensor = torch::from_blob(input.data, {kImageHeight, kImageWidth,kChannels}, torch::kFloat32);
	input_tensor.toType(torch::kFloat32);
	input_tensor = input_tensor.unsqueeze(0);
	std::cout << input_tensor.sizes() << std::endl;
	input_tensor = input_tensor.permute({0,3,1,2});
	std::cout << input_tensor.sizes() << std::endl;
	input_tensor = input_tensor.to(at::kCUDA);
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(input_tensor);
	auto outputs = model_->forward(inputs).toTuple();
	torch::Tensor pred = outputs->elements()[0].toTensor();
	std::cout << pred.sizes()<< endl;
	result = DetectionLayer(pred, cls_scores);
	scale_loc(result, input, image);
	for ( int i = 0; i < result.size(); i++) {
		cv::rectangle(pic, cv::Point(result[i][0], result[i][1]), cv::Point(result[i][2], result[i][3]), cv::Scalar(255, 255, 0), 2);
		int class_index = result[i][4];
		cv::putText(pic, classnames_[class_index] + to_string(cls_scores[i]).substr(0, 4),
			cv::Point(result[i][0], result[i][1] -20), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 255, 0),
			2, 8, 0);
	}
	cv::imwrite("./final_result.jpg",pic);
    return result;
}