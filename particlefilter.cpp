#include "particlefilter.h"

#include <iostream>
using namespace std;



void ComputeHist(Mat& src, Mat& hist)
{
Mat hsv;
cvtColor(src, hsv, COLOR_BGR2HSV);

int h_bins = 50;
int s_bins = 32;
int v_bins = 10;
int histSize[] = { h_bins, s_bins, v_bins };

float h_ranges[] = {0, 180};
float s_ranges[] = {0, 255};
float v_ranges[] = { 0, 255};
const float* ranges[] = {h_ranges, s_ranges, v_ranges};

int channels[] = { 0, 1, 2};

calcHist(&hsv, 1, channels, cv::noArray(), hist, 3, histSize, ranges, true);
}


double CalculateWeight(Mat& src, const Mat* hist)
{
Mat histsrc;
ComputeHist(src, histsrc);
return compareHist(*hist, histsrc, CV_COMP_BHATTACHARYYA);
}


particleFilter::particleFilter(Mat& HistOfObject)
{
    ObjectHisto = &HistOfObject;

    nParticles = 0;
    gsl_rng_env_setup();
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, time(NULL));

}

particleFilter::~particleFilter() {}

void particleFilter::initParticles(Rect region, int particlesPerObject)
{
     nParticles = particlesPerObject;

     int width  = region.width;
     int height = region.height;
     int xinit = (float) region.x + width/2;
     int yinit = (float) region.y + height/2;

     for ( int i=0; i < nParticles; i++ )
         {
           particles[i].x0 = particles[i].xp = particles[i].x = xinit;
           particles[i].y0 = particles[i].yp = particles[i].y = yinit;
           particles[i].sp = particles[i].s = 0.5;
           particles[i].w = 0;
           particles[i].width = width;
           particles[i].height = height;
         }
}

Rect particleFilter::getParticleRect()
{
    Rect rect;
    rect.x = cvRound(particles[0].x - 0.5 * particles[0].s * particles[0].width);
    rect.y = cvRound(particles[0].y - 0.5 * particles[0].s * particles[0].height);
    rect.width = cvRound(particles[0].s * particles[0].width);
    rect.height = cvRound(particles[0].s * particles[0].height);

    return rect;
}

CvPoint particleFilter::getParticleCenter()
{
    return cvPoint(cvRound(particles[0].x), cvRound(particles[0].y));
}



void particleFilter::transition(Mat& frame, int w, int h)
{
    for ( int i=0; i < nParticles; i++)
        {
          particles[i] = calTransition(particles[i], frame, w, h, rng);
        }
}

void particleFilter::normalizeWeights()
{
    float sum = 0;
    for ( int i=0; i < nParticles; i++ )
        {
          sum += particles[i].w;
        }

    for ( int i=0; i < nParticles; i++ )
        {
          particles[i].w /= sum;
        }
}

int particleCmp(const void* p1, const void* p2)
{
    particle* _p1 = (particle*)p1;
    particle* _p2 = (particle*)p2;

    if( _p1->w > _p2->w )
        return 1;
    if( _p1->w < _p2->w )
        return -1;
    return 0;
}

void particleFilter::resample()
{
    int np, k = 0;
    particle * newParticles;
    qsort(particles, nParticles, sizeof(particle), &particleCmp);
    newParticles = (particle*) malloc(nParticles * sizeof(particle));
    for (int i=0; i < nParticles; i++)
        {
        np = cvRound(particles[i].w * nParticles);
        for (int j=0; j < np; j++)
            {
            newParticles[k++] = particles[i];
            if (k == nParticles)
               {
                 goto exit;
               }
            }
        }

    while (k < nParticles)
          {
            newParticles[k++] = particles[0];
          }

    exit:
    for ( int i=0; i < nParticles; i++ )
        {
          particles[i] = newParticles[i];
        }
    free(newParticles);
}

void particleFilter::displayParticles(Mat& img, CvScalar nColor, CvScalar hColor, int param)
{
    CvScalar color;
    if (param == SHOW_ALL)
    {
        for (int i=nParticles-1; i >= 0; i--)
            {
            if (i == 0)
               {
                 color = hColor;
               } else
                     {
                       color = nColor;
                     }
            displayParticle(img, particles[i], color);
            }
    } else if ( param == SHOW_SELECTED )
              {
                color = hColor;
                displayParticle(img, particles[0], color);
              }
}


particle particleFilter::calTransition(particle p, Mat& frame, int w, int h, gsl_rng* rng)
{
         particle pn;
         particle pv;
         pv.w = 1;

         int counter = 0;
         do
         {
         double x = A1 * (p.x - p.x0) + A2 * (p.xp - p.x0) + B0 * gsl_ran_gaussian(rng, TRANS_X_STD) + p.x0;
         double y = A1 * (p.y - p.y0) + A2 * (p.yp - p.y0) + B0 * gsl_ran_gaussian(rng, TRANS_Y_STD) + p.y0;

         pn.x = (float) MAX( 0.0, MIN((float) w - 1.0, x));
         pn.y = (float) MAX( 0.0, MIN((float) h - 1.0, y));

         pn.s = 1.0;
         pn.xp = p.x;
         pn.yp = p.y;
         pn.sp = p.s;
         pn.x0 = p.x0;
         pn.y0 = p.y0;
         pn.width = p.width;
         pn.height = p.height;
         pn.w = 0;

         float s = pn.s;
         int rr = cvRound(pn.y);
         int cc = cvRound(pn.x);
         int ww = cvRound(pn.width * s);
         int hh = cvRound(pn.height * s);
         CvRect MyRect = cvRect(cc-ww/2, rr - hh/2, ww, hh);
         if (   ((cc+ww/2)<frame.cols-1) and ((rr+hh/2)<frame.rows-1) and ((cc-ww/2) > 0) and ((rr-hh/2) > 0) )
         {
            Mat src = frame(MyRect);
            pn.w = CalculateWeight(src, ObjectHisto);
         }
         else
         {
            pn.w = 1;
         }
         if( (particleCmp(&pn,&pv)) == -1 ){pv=pn;}

         counter++;
         } while (counter<5);

         return pv;
}

void particleFilter::displayParticle(Mat& img, particle p, CvScalar color)
{
     int x0 = cvRound( p.x - 0.5 * p.s * p.width );
     int y0 = cvRound( p.y - 0.5 * p.s * p.height );
     int x1 = x0 + cvRound( p.s * p.width );
     int y1 = y0 + cvRound( p.s * p.height );

     rectangle( img, cvPoint( x0, y0 ), cvPoint( x1, y1 ), color, 2, 8, 0 );
}
