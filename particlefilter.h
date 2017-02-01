#ifndef PARTICLEFILTER_H_INCLUDED
#define PARTICLEFILTER_H_INCLUDED

#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>

#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

//#define TRANS_X_STD 0.5
//#define TRANS_Y_STD 1.0

#define TRANS_X_STD 1.0
#define TRANS_Y_STD 1.0


#define TRANS_S_STD 0.001
#define MAX_PARTICLES 150

#define SHOW_ALL 0
#define SHOW_SELECTED 1

#define A1 2.0
#define A2 -1.0
#define B0 1.0000

using namespace cv;


typedef struct particle {
       float x; /** current x coordinate */
       float y; /** current y coordinate */
       float w; /** weight*/

       float s; /** scale */
       float xp; /** previous x coordinate */
       float yp; /** previous y coordinate */
       float sp; /** previous scale */
       float x0; /** original x coordinate */
       float y0; /** original y coordinate */
       int width; /** original width of region described by particle */
       int height; /** original height of region described by particle */

                        } particle;



void   ComputeHist(Mat& src, Mat& hist);


double CalculateWeight(Mat& src,const Mat* hist);




class particleFilter {
      public:
            particle particles[MAX_PARTICLES];
            int nParticles;
            gsl_rng* rng;
            float weight;
            Rect region;

            const Mat* ObjectHisto;

            particleFilter(Mat& HistOfObject);
           ~particleFilter();
            // Initializes particles //
            void initParticles(Rect region, int particlesPerObject);
            Rect getParticleRect();
            CvPoint getParticleCenter();


            /** Moves particles */
            void transition(Mat& frame, int w, int h);
            /** Normalize weights of particles */
            void normalizeWeights();
            /** Resamples particles */
            void resample();
            void displayParticles(Mat& img, CvScalar nColor, CvScalar hColor, int param);


      private:
             particle calTransition(particle p, Mat& frame, int w, int h, gsl_rng* rng);
             void displayParticle(Mat& img, particle p, CvScalar color);
                        };


#endif // PARTICLEFILTER_H_INCLUDED
