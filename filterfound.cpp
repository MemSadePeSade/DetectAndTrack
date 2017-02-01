#include "filterfound.h"

using namespace cv::ximgproc;
using namespace cv::ml;


int   upgradeDPM(Mat &frame, vector<Rect> &found, vector<double> &foundweigths, int foundsize)
{


Ptr<SuperpixelSEEDS> seeds;

Mat frame_roi(Size(128,64), CV_8UC3, cv::Scalar(0,0,0));
Mat labels;

int   *NumSP   = NULL;
float *VecFeat = NULL;
NumSP = new int [foundsize] ;
if(NumSP==NULL)return -1;
VecFeat = new float [foundsize*5*num_superpixels];
if(VecFeat==NULL)return -1;






int NumVec = 0 ;

for (size_t i = 0; i<found.size(); i++ ){

     cv::resize(frame(found[i]), frame_roi, frame_roi.size());
     cvtColor(frame_roi, frame_roi, COLOR_BGR2Lab);
     seeds = createSuperpixelSEEDS(frame_roi.cols, frame_roi.rows, frame_roi.channels(), num_superpixels,
                                   num_levels, prior, num_histogram_bins, double_step);
     seeds->iterate(frame_roi, num_iterations);
     seeds->getLabels(labels);

     NumSP[i] = seeds->getNumberOfSuperpixels();


     ExtFFSP(labels, frame_roi, VecFeat+5*NumVec, NumSP[i]);
     NumVec+=NumSP[i];
    }





Mat MatrixForSVM(foundsize, ClusterCount, CV_32F);
HistForSVM(MatrixForSVM, VecFeat, NumSP, NumVec, foundsize);
MatrixForSVM.convertTo(MatrixForSVM, CV_32F);
DetectSVM(foundweigths, MatrixForSVM, foundsize);


delete [] NumSP;
delete [] VecFeat;

return 0;
}



void   ExtFFSP(Mat &labels, Mat &frame_roi, float *iter,int NumSP)
{
int NumLabels[NumSP] = {0};
Size s  = labels.size() ;
int sheight = s.height;
int swidth  = s.width;


for (int i;i<5*(NumSP);i++){*(iter+i)=0;}


for (int i=0; i<sheight; i++)
    {
    for (int j=0; j<swidth; j++)
        {
         NumLabels[labels.at<int>(i,j)]++;
         *(iter+5*labels.at<int>(i,j)+0) += float(i)/sheight;
         *(iter+5*labels.at<int>(i,j)+1) += float(j)/swidth;
         *(iter+5*labels.at<int>(i,j)+2) += float(frame_roi.at<Vec3b>(i,j)[0])/255;
         *(iter+5*labels.at<int>(i,j)+3) += float(frame_roi.at<Vec3b>(i,j)[1])/255;
         *(iter+5*labels.at<int>(i,j)+4) += float(frame_roi.at<Vec3b>(i,j)[2])/255;
        }
    }


for (int i=0; i<NumSP; i++)
    {
    for (int j=0; j<5; j++)
        {
        *(iter+5*i+j)/=NumLabels[i];
        }
    }


}



void   HistForSVM(Mat &MatrixForSVM, float *VecFeat, int *NumSP, int NumVec, int foundsize)
{
Mat points(NumVec,5, CV_32F, VecFeat);
Mat labels(NumVec,1, CV_32S);


kmeans(points, ClusterCount, labels,
       TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
       5, KMEANS_PP_CENTERS , noArray());


int ind1=0; int ind2=0;
for (int i=0; i<foundsize; i++)
{
     float hist[ClusterCount] = {0};
     ind2+=NumSP[i];
     for(int  j=ind1;j<ind2;j++)
        {
         hist[int(labels.at<int>(j))]++;
        }
ind1=ind2;
Mat row(1, ClusterCount, CV_32F,&hist[0]);

row.row(0).copyTo(MatrixForSVM.row(i));

}
}



int DetectSVM(vector<double> &foundweights,Mat &MatrixForSVM, int foundsize)
{
Mat DetectData;
Mat TrainData;
Mat labelssvm(0, 1, CV_32S);

vector<int> indexpermutate1;
vector<int> indexpermutate2;



for(int i=0; i<foundsize; i++ )
         {
         const double  tp1= 1.3; const double tn1 = 0.5 ;
         if (foundweights[i] > tp1)
             {
              indexpermutate1.push_back(i);
              labelssvm.push_back(1);
              TrainData.push_back(MatrixForSVM.row(i));
             }
             if (tp1 > foundweights[i]  and foundweights[i] > tn1)
             {
              indexpermutate2.push_back(i);
              DetectData.push_back(MatrixForSVM.row(i));
             }
             if (foundweights[i] < tn1)
             {
              indexpermutate1.push_back(i);
              labelssvm.push_back(-1);
              TrainData.push_back(MatrixForSVM.row(i));
             }
         }

for (int i=0;i<indexpermutate1.size();i++){foundweights[indexpermutate1[i]]=labelssvm.at<int>(i);}

int ind1 = 0;
int ind2 = labelssvm.rows;
int schet=0;
Mat res;   // output

if(DetectData.empty() == true){cout << "no detectdata" << endl;return 0;}

while(ind1!=ind2 ){
schet++;
cout << "$$$$$$$$$$$$" << endl;
ind1 = ind2;
Ptr<ml::SVM> svm = ml::SVM::create();
svm->setType(ml::SVM::C_SVC);
svm->setKernel(ml::SVM::LINEAR);
svm->train(TrainData , ml::ROW_SAMPLE , labelssvm );
svm->predict(DetectData, res);

Mat supportvector = svm->getSupportVectors();
if(supportvector.empty() == true){cout << "no two classes" << endl;break;}




Mat alpha, svidx;
double rho = svm->getDecisionFunction(0, alpha, svidx);



Mat distance(1, DetectData.rows, CV_64F, cv::Scalar(0) );
Mat tempDetectData;
vector<int> tempindexpermutate2;



for (int i=0;i<DetectData.rows;i++)
{
distance.at<double>(i) += rho;
for(int j=0;j < supportvector.rows; j++)
                 {
                 Mat tempsupportvector = supportvector.row(j);
                 distance.at<double>(i) += tempsupportvector.dot( DetectData.row(i) ) ;
                 }



const double  tp2= 0.5; const double tn2 = -1.0 ;
if ( distance.at<double>(i) < tn2 or distance.at<double>(i) > tp2)
          {
           indexpermutate1.push_back(indexpermutate2[i]);
           labelssvm.push_back(res.at<int>(i));
           TrainData.push_back(DetectData.row(i));
          }
else      {
           tempindexpermutate2.push_back(indexpermutate2[i]);
           tempDetectData.push_back(DetectData.row(i));
          }
}

cout << "distance= "<< endl << " "  << distance << endl << endl;

indexpermutate2.clear();
indexpermutate2 = tempindexpermutate2;
tempindexpermutate2.clear();



if(tempDetectData.empty() == true){cout << "vse rasklasific" << endl;break;}


DetectData.release();
DetectData = Mat(tempDetectData);
tempDetectData.release();
ind2 = labelssvm.rows;





for (int i=0;i<indexpermutate1.size();i++){foundweights[indexpermutate1[i]]=labelssvm.at<int>(i);}
for (int i=0;i<indexpermutate2.size();i++){foundweights[indexpermutate2[i]]=res.at<int>(i);}

supportvector.release();
res.release();
                     }



return 1;
}


