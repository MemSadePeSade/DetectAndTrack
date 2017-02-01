#ifndef PEOPLEDETECT__H_INCLUDED
#define PEOPLEDETECT__H_INCLUDED

#include <stdio.h>

#include "filterfound.h"
#include "opencv2/cudaobjdetect.hpp"

#include <opencv2/core/utility.hpp>
#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/imgproc.hpp"

#define win_size     Size(64, 128)
#define block_size   Size(16, 16)
#define block_stride Size(8, 8)
#define cell_size    Size(8, 8)
#define nbins        9

#define nlevels         100
#define hit_threshold   0
#define win_stride      Size(16,16)
#define scale           1.05
#define gr_threshold     0


using namespace cv;
using namespace std;
int peopledetect(Mat& frame, Mat& FrameHogDescriptors, vector<Rect>& found_filtered2);

#endif // PEOPLEDETECT__H_INCLUDED
