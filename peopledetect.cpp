#include "peopledetect.h"




int peopledetect(Mat &frame, Mat &FrameHogDescriptors, vector<Rect> &found_filtered2)
{
vector<Rect> found;
vector<double> foundweights;

cuda::GpuMat gpu_img;
cuda::GpuMat descriptors;

cv::Ptr<cv::cuda::HOG> gpu_hog = cv::cuda::HOG::create(win_size, block_size, block_stride, cell_size, nbins);
gpu_hog->setSVMDetector(gpu_hog->getDefaultPeopleDetector());

Mat img_aux;
cvtColor(frame, img_aux, COLOR_BGR2BGRA);

gpu_img.upload(img_aux);
gpu_hog->setNumLevels(nlevels);
gpu_hog->setHitThreshold(hit_threshold);
gpu_hog->setWinStride(win_stride);
gpu_hog->setScaleFactor(scale);
gpu_hog->setGroupThreshold(gr_threshold);

double t = (double)getTickCount();

gpu_hog->compute(gpu_img, descriptors, cuda::Stream::Null());
gpu_hog->detectMultiScale(gpu_img, found, &foundweights);


descriptors.download(FrameHogDescriptors);


t = (double)getTickCount() - t;
printf("tdetection time = %gms\n", t*1000./cv::getTickFrequency());


for(size_t i=0; i<found.size(); i++ )
   {
           Rect r = found[i];
           r.x += cvRound(r.width*0.1);
           r.width = cvRound(r.width*0.8);
           r.y += cvRound(r.height*0.07);
           r.height = cvRound(r.height*0.8);
           found[i] = r;
    }





if(!foundweights.empty()){ upgradeDPM( frame, found, foundweights, found.size() ); }



vector<Rect> found_filtered1;
for (size_t i=0; i<found.size(); i++ )
    {
            if(foundweights[i] == 1)
              {
                 Rect r = found[i];
                 found_filtered1.push_back(r);
              }
    }


found_filtered2 = found_filtered1;
size_t i,j;
for (i=0; i<found_filtered1.size(); i++)
    {
            Rect r = found_filtered1[i];
            for (j=0; j<found_filtered1.size(); j++)
                if (j!=i && (r & found_filtered1[j]) == r)
                    break;
            if (j== found_filtered1.size())
                found_filtered2.push_back(r);
    }



return 0;
}
