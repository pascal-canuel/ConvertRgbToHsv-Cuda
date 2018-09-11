// MulMat.cpp : définit le point d'entrée pour l'application console.
//

#include "stdafx.h"
#include <string>
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h" 
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp> 
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/types_c.h> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <cmath>

using namespace cv;

extern "C" bool GPGPU_TstImg_CV_8U(cv::Mat* img, cv::Mat* GPGPUimg);

Mat imgSobelCPU;

int main()
{
	String imgPath = "../picture/1.jpg";
	Mat imgInput = imread(imgPath);
	Mat imgOutput = imread(imgPath);
	
	imshow("lenaInput", imgInput);

	GPGPU_TstImg_CV_8U(&imgInput, &imgOutput);

	imshow("lenaOutput", imgOutput);

	//	APPLY GAUSSIAN BLUR TO REMOVE NOISE

	////	CONVERT RGB TO HSV
	//String imgPathRGB = "../picture/1.jpg";
	//Mat imgRGB = imread(imgPathRGB);

	//imshow("1_RGB", imgRGB);

	//for (int y = 0; y < imgRGB.rows; y++) {
	//	for (int x = 0; x < imgRGB.cols; x++) {
	//		Vec3b pix = imgRGB.at<Vec3b>(y, x);

	//		double blue = (double)pix.val[0] / 255;
	//		double green = (double)pix.val[1] / 255;
	//		double red = (double)pix.val[2] / 255;

	//		double cMax = max(max(blue, green), red);
	//		/*if ((blue >= green) && (blue >= red))
	//			cMax = blue;
	//		else if ((green >= blue) && (green >= red))
	//			cMax = green;
	//		else
	//			cMax = red;*/

	//		double cMin = min(min(blue, green), red);
	//		/*if ((blue <= green) && (blue <= red))
	//			cMin = blue;
	//		else if ((green <= blue) && (green <= red))
	//			cMin = green;
	//		else
	//			cMin = red;*/
	//		
	//		double delta = cMax - cMin;

	//		//	HUE
	//		double hue = 0;
	//		
	//		if (blue == cMax) {
	//			hue = 60 * ((red - green) / delta + 4);
	//		}
	//		else if (green == cMax) {
	//			hue = 60 * ((blue - red) / delta + 2);
	//		}
	//		else if (red == cMax) {
	//			hue = 60 * ((green - blue) / delta);	
	//			if (hue < 0)
	//				hue += 360;
	//		}

	//		//	SATURATION
	//		double saturation = 0;
	//		if (cMax != 0) {
	//			saturation = delta / cMax;
	//		}

	//		//	VALUE
	//		double value = cMax;

	//		//	MAP 0-100% TO OPENCV 180:255:255
	//		imgRGB.at<Vec3b>(y, x)[0] = hue / 2;
	//		imgRGB.at<Vec3b>(y, x)[1] = saturation * 255;
	//		imgRGB.at<Vec3b>(y, x)[2] = value * 255;
	//	}
	//}

	//imshow("1_HSV", imgRGB);


	//Mat imgHSV = imread(imgPathRGB);
	//cvtColor(imgHSV, imgHSV, CV_BGR2HSV);
	//imshow("OPENCV_HSV", imgHSV);

	waitKey(0);
}