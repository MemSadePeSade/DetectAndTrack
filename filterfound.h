#ifndef FILTERFOUND_H
#define FILTERFOUND_H


#include <stdio.h>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml.hpp"
#include <opencv2/ximgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
///////////////////////////parameters for SuperpixelSEEDS///////////////////////////////
#define num_iterations  4
#define prior  2
#define double_step  false
#define num_superpixels  125
#define num_levels  4
#define num_histogram_bins  5
///////////////////////////parameters for SuperpixelSEEDS///////////////////////////////
#define ClusterCount  100
////////////////////////////////////////////////////////





using namespace cv;
using namespace std;



int    upgradeDPM(Mat &frame, vector<Rect> &found, vector<double> &foundweigths,int foundsize);
void   ExtFFSP(Mat &labels, Mat &frame_roi, float *iter, int NumSP);
void   HistForSVM(Mat &MatrixForSVM, float *VecFeat, int *NumSP, int NumVec, int foundsize);
int    DetectSVM(vector<double> &foundweigths,Mat &MatrixForSVM, int foundsize);


#endif
