#include "peopledetect.h"
#include "particlefilter.h"


int main(int argc, char** argv)
{
    if( argc == 1 )
    {
        printf("Usage: peopledetect (<image_filename>)\n");
        return 0;
    }

    VideoCapture cap(argv[1]);

    if(!cap.isOpened())
      {
        fprintf( stderr, "ERROR: the specified file could not be loaded\n");
        return -1;
      }


    cvNamedWindow( "PlayVideo", CV_WINDOW_AUTOSIZE );
    Mat frame;
    Mat FrameHogDescriptors;

    vector<Rect> found;
    vector<particleFilter> VectorTrackers;
    vector<Mat> HistogramsOfObjects(10);

    int Triger=0;
    const Mat hist;

    VideoWriter outputVideo;
    const char *filename = "capture.avi";
    double fps = 15;
    CvSize size = cvSize(1280, 960);

    VideoWriter writer;
    writer.open( filename, CV_FOURCC('M','J','P','G'), fps, size );

    while(cap.read(frame))
    {
         if ( Triger == 0 )
            {
              peopledetect(frame, FrameHogDescriptors, found);
              for( size_t i=0; i<found.size(); i++ )
                 {
                   Mat src = frame(found[i]);
                   ComputeHist(src, HistogramsOfObjects[i]);
                   Rect r = found[i];
                   VectorTrackers.push_back(particleFilter(HistogramsOfObjects[i]));
                   VectorTrackers[i].initParticles( found[i], 25);
                   Triger = 1;
                   rectangle(frame, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
                 }
            }

         if ( Triger == 1 )
         {
            for( size_t i=0; i<found.size(); i++ )
               {
                 VectorTrackers[i].transition(frame, frame.cols, frame.rows);
                 VectorTrackers[i].normalizeWeights();
                 VectorTrackers[i].resample();
                 VectorTrackers[i].displayParticles(frame, CV_RGB(0, 0, 255), CV_RGB(0, 0, 255), 1);
               }
          }

         writer << frame;
         imshow("PlayVideo", frame);
         int  c = waitKey(1) ;
         if ( c == 32)
         {
            break;
         }

    }

    return 3;
}
