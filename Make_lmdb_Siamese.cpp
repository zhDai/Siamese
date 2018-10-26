#include <direct.h>
#include <io.h>
#include "glog/logging.h"
#include "google/protobuf/text_format.h"//解析proto类型文件中，解析prototxt类型的头文件
#include "leveldb/db.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

//大小端转换。mnist原始数据文件中32为整型值为大端存储，c/c++的微小端存储
uint32_t swap_endian(uint32_t val) {
	//<<为位操作符，“<<”左移一位，实际数值乘以2，整形数字2，对应二进制为：……010，2<<2 ……01000，左移两位后，变成8
	//变量之间的“&”为按照“位”，进行与操作，二进制数：1010 & 0110 =0010
	// 变量之间的“|”操作符为按照“位”进行或操作，二进制数：1010 & 0110 =1110
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}

void read_image(std::ifstream* image_file, std::ifstream* label_file,
	uint32_t index, uint32_t rows, uint32_t cols,
	char* pixels, char* label) {
	//seekg（）是对输入文件定位，它有两个参数：第一个参数是偏移量，第二个参数是基地址。
	image_file->seekg(index * rows * cols + 16);
	image_file->read(pixels, rows * cols);
	label_file->seekg(index + 8);
	label_file->read(label, 1);
}

void convert_dataset(const char* image_filename, const char* label_filename,
	const char* db_filename) {
	// 用c++输入文件流以二进制方式打开文件
	std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
	std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
	CHECK(image_file) << "Unable to open file " << image_filename;
	CHECK(label_file) << "Unable to open file " << label_filename;
	// 读取 the magic and the meta data
	uint32_t magic;
	uint32_t num_items;
	uint32_t num_labels;
	uint32_t rows;
	uint32_t cols;
	/*reinterpret_cast
	从指针类型到一个足够大的整数类型
	从整数类型或者枚举类型到指针类型
	从一个指向函数的指针到另一个不同类型的指向函数的指针
	从一个指向对象的指针到另一个不同类型的指向对象的指针
	从一个指向类函数成员的指针到另一个指向不同类型的函数成员的指针
	从一个指向类数据成员的指针到另一个指向不同类型的数据成员的指针
	*/
	image_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	CHECK_EQ(magic, 2051) << "Incorrect image file magic.";
	label_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	CHECK_EQ(magic, 2049) << "Incorrect label file magic.";
	image_file.read(reinterpret_cast<char*>(&num_items), 4);
	num_items = swap_endian(num_items);
	label_file.read(reinterpret_cast<char*>(&num_labels), 4);
	num_labels = swap_endian(num_labels);
	CHECK_EQ(num_items, num_labels);
	image_file.read(reinterpret_cast<char*>(&rows), 4);
	rows = swap_endian(rows);
	image_file.read(reinterpret_cast<char*>(&cols), 4);
	cols = swap_endian(cols);

	// 打开 leveldb，创建leveldb类型的指针  ，其实是创建数据库
	leveldb::DB* db;
	//Options打开leveldb文件的方式，类似这种“存在就打开，不存在就创建”的文件打开
	//方式，创建对数据库的操作对象，通过options来对db做操作。
	leveldb::Options options;
	options.create_if_missing = true;// 存在就报错  
	options.error_if_exists = true;// 不存在就创建 
	leveldb::Status status = leveldb::DB::Open(
		options, db_filename, &db);
	CHECK(status.ok()) << "Failed to open leveldb " << db_filename
		<< ". Is it already existing?";

	char label_i;
	char label_j;
	//定义char指针，指向字符串数组，字符串数组的容量为一个图片的大小 
	char* pixels = new char[2 * rows * cols];
	const int kMaxKeyLength = 10;//最大的键值长度
	char key[kMaxKeyLength];
	std::string value;
	//设置datum数据对象的结构，其结构和源图像结构相同
	caffe::Datum datum;
	// 在一对图像中，每个图像都是一个通道
	datum.set_channels(2);
	datum.set_height(rows);
	datum.set_width(cols);
	LOG(INFO) << "A total of " << num_items << " items.";
	LOG(INFO) << "Rows: " << rows << " Cols: " << cols;
	for (int itemid = 0; itemid < num_items; ++itemid) {
		int i = caffe::caffe_rng_rand() % num_items;  // pick a random  pair
		int j = caffe::caffe_rng_rand() % num_items;
		read_image(&image_file, &label_file, i, rows, cols,
			pixels, &label_i);
		read_image(&image_file, &label_file, j, rows, cols,
			pixels + (rows * cols), &label_j);
		datum.set_data(pixels, 2 * rows*cols);
		if (label_i == label_j) {
			datum.set_label(1);
		}
		else {
			datum.set_label(0);
		}
		datum.SerializeToString(&value);
		snprintf(key, kMaxKeyLength, "%08d", itemid);
		db->Put(leveldb::WriteOptions(), std::string(key), value);
	}

	delete db;
	delete pixels;
}
/*
Argc为统计main函数接受的参数个数，正常调用时argc=4，argv为对应的参数值，
argv[0]=可执行文件名，argv[1]=源数据路径，arg[2]=标签数据路径，
arg[3]=保存lmdb数据的路径
*/
int main(int argc, char** argv) {
	if (argc != 4) {
		printf("This script converts the MNIST dataset to the leveldb format used\n"
			"by caffe to train a siamese network.\n"
			"Usage:\n"
			"    convert_mnist_data input_image_file input_label_file "
			"output_db_file\n"
			"The MNIST dataset could be downloaded at\n"
			"    https://yann.lecun.com/exdb/mnist/\n"
			"You should gunzip them after downloading.\n");
	}
	else {
		google::InitGoogleLogging(argv[0]);
		convert_dataset(argv[1], argv[2], argv[3]);
	}
	return 0;
}

