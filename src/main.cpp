#include "yolo.hpp"
using namespace std;

int main() {
	string model_path = "D:/test/model/model.pt";
	torch::DeviceType device_type;
	if (torch::cuda::is_available) {
		cout << "cuda is ok" << endl;
		device_type = torch::kCUDA;
	}
	else device_type = torch::kCPU;
	cout << device_type << endl;
	torch::Device device(device_type);
	YOLOV3SPP yolov3spp;
	yolov3spp.Initialize(&device);
	cv::Mat image = cv::imread("D:/code/libtorch_yolov3/image/4070.jpg");
	yolov3spp.Forward(image);
	auto a = torch::arange(15).view({ 5,3 });
	auto b = torch::arange(5).view({ 5 , 1 });
	auto c = torch::cat({a, b}, 1);
	std::cout << c << endl;
	//a.slice(1, 5, 16) = a.slice(1, 5, 16) * 2;
	//cout << a << endl;
	//auto index = ((a.slice(1, 2, 4) > 32)).all(1);
	//cout << index << endl;
	//auto mask_inx = torch::nonzero(index).squeeze(1);
	//cout << mask_inx << endl;
 //   a = torch::index_select(a, 0, mask_inx);
	//cout << a << endl;
}