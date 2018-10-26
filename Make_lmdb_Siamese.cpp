#include <direct.h>
#include <io.h>
#include "glog/logging.h"
#include "google/protobuf/text_format.h"//����proto�����ļ��У�����prototxt���͵�ͷ�ļ�
#include "leveldb/db.h"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

//��С��ת����mnistԭʼ�����ļ���32Ϊ����ֵΪ��˴洢��c/c++��΢С�˴洢
uint32_t swap_endian(uint32_t val) {
	//<<Ϊλ����������<<������һλ��ʵ����ֵ����2����������2����Ӧ������Ϊ������010��2<<2 ����01000��������λ�󣬱��8
	//����֮��ġ�&��Ϊ���ա�λ�������������������������1010 & 0110 =0010
	// ����֮��ġ�|��������Ϊ���ա�λ�����л����������������1010 & 0110 =1110
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}

void read_image(std::ifstream* image_file, std::ifstream* label_file,
	uint32_t index, uint32_t rows, uint32_t cols,
	char* pixels, char* label) {
	//seekg�����Ƕ������ļ���λ������������������һ��������ƫ�������ڶ��������ǻ���ַ��
	image_file->seekg(index * rows * cols + 16);
	image_file->read(pixels, rows * cols);
	label_file->seekg(index + 8);
	label_file->read(label, 1);
}

void convert_dataset(const char* image_filename, const char* label_filename,
	const char* db_filename) {
	// ��c++�����ļ����Զ����Ʒ�ʽ���ļ�
	std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
	std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
	CHECK(image_file) << "Unable to open file " << image_filename;
	CHECK(label_file) << "Unable to open file " << label_filename;
	// ��ȡ the magic and the meta data
	uint32_t magic;
	uint32_t num_items;
	uint32_t num_labels;
	uint32_t rows;
	uint32_t cols;
	/*reinterpret_cast
	��ָ�����͵�һ���㹻�����������
	���������ͻ���ö�����͵�ָ������
	��һ��ָ������ָ�뵽��һ����ͬ���͵�ָ������ָ��
	��һ��ָ������ָ�뵽��һ����ͬ���͵�ָ������ָ��
	��һ��ָ���ຯ����Ա��ָ�뵽��һ��ָ��ͬ���͵ĺ�����Ա��ָ��
	��һ��ָ�������ݳ�Ա��ָ�뵽��һ��ָ��ͬ���͵����ݳ�Ա��ָ��
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

	// �� leveldb������leveldb���͵�ָ��  ����ʵ�Ǵ������ݿ�
	leveldb::DB* db;
	//Options��leveldb�ļ��ķ�ʽ���������֡����ھʹ򿪣������ھʹ��������ļ���
	//��ʽ�����������ݿ�Ĳ�������ͨ��options����db��������
	leveldb::Options options;
	options.create_if_missing = true;// ���ھͱ���  
	options.error_if_exists = true;// �����ھʹ��� 
	leveldb::Status status = leveldb::DB::Open(
		options, db_filename, &db);
	CHECK(status.ok()) << "Failed to open leveldb " << db_filename
		<< ". Is it already existing?";

	char label_i;
	char label_j;
	//����charָ�룬ָ���ַ������飬�ַ������������Ϊһ��ͼƬ�Ĵ�С 
	char* pixels = new char[2 * rows * cols];
	const int kMaxKeyLength = 10;//���ļ�ֵ����
	char key[kMaxKeyLength];
	std::string value;
	//����datum���ݶ���Ľṹ����ṹ��Դͼ��ṹ��ͬ
	caffe::Datum datum;
	// ��һ��ͼ���У�ÿ��ͼ����һ��ͨ��
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
ArgcΪͳ��main�������ܵĲ�����������������ʱargc=4��argvΪ��Ӧ�Ĳ���ֵ��
argv[0]=��ִ���ļ�����argv[1]=Դ����·����arg[2]=��ǩ����·����
arg[3]=����lmdb���ݵ�·��
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

