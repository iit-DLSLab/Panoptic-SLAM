/*
* This file is a modified version of ORB-SLAM3.<https://github.com/UZ-SLAMLab/ORB_SLAM3>
*
* This file is part of Panoptic-SLAM.
* Copyright (C) 2024 Gabriel Fischer Abati & João Carlos Virgolino Soares 
* Istituto Italiano di Tecnologia
* Pontifical Catholic University of Rio de Janeiro - PUC-Rio
* Co-authors: Vivian Suzano Medeiros, Marco Antonio Meggiolaro and Claudio Semini
* Please report suggestions and comments to gabriel.fischer@iit.it
*/


#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include "Thirdparty/Sophus/sophus/geometry.hpp"

#include "ImuTypes.h"
#include "ORBVocabulary.h"


#include "Converter.h"
#include "Settings.h"

#include <mutex>
#include <opencv2/opencv.hpp>

#include "Eigen/Core"
#include "sophus/se3.hpp"
#include "Panoptic_Struct.h"


namespace ORB_SLAM3
{
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;
class ConstraintPoseImu;
class GeometricCamera;
class ORBextractor;


class Frame
{
public:
    Frame();

    // Copy constructor.
    Frame(const Frame &frame);

    // Constructor for stereo cameras.
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

    // Constructor for RGB-D cameras.
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

    // Constructor for Monocular cameras.
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

    // Destructor
    // ~Frame();

    // Extract ORB on the image. 0 for left image and 1 for right image.
    void ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1);

    // Compute Bag of Words representation.
    void ComputeBoW();

    // Set the camera pose. (Imu pose is not modified!)
    void SetPose(const Sophus::SE3<float> &Tcw);

    // Set IMU velocity
    void SetVelocity(Eigen::Vector3f Vw);

    Eigen::Vector3f GetVelocity() const;

    // Set IMU pose and velocity (implicitly changes camera pose)
    void SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb);

    Eigen::Matrix<float,3,1> GetImuPosition() const;
    Eigen::Matrix<float,3,3> GetImuRotation();
    Sophus::SE3<float> GetImuPose();

    Sophus::SE3f GetRelativePoseTrl();
    Sophus::SE3f GetRelativePoseTlr();
    Eigen::Matrix3f GetRelativePoseTlr_rotation();
    Eigen::Vector3f GetRelativePoseTlr_translation();

    void SetNewBias(const IMU::Bias &b);

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    bool ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v);

    Eigen::Vector3f inRefCoordinates(Eigen::Vector3f pCw);

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1, const bool bRight = false) const;

    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    void ComputeStereoMatches();

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    bool UnprojectStereo(const int &i, Eigen::Vector3f &x3D);

    void Set_Panoptic_Image(cv::Mat panoptic);
    cv::Mat panoptic_image;

    cv::Mat stuff_descriptor;
    std::map<int,cv::Mat> thing_descriptor;
    cv::Mat unknown_descriptor;
    cv::Mat all_masks;
    std::vector<cv::KeyPoint> stuff_keypoints;
    std::map<int,std::vector<cv::KeyPoint>> thing_keypoints;
    std::vector<cv::KeyPoint> unknown_keypoints;

    std::vector<Panoptic_Object> Panoptic_objs;
    void set_stuff_keypoints(std::vector<cv::KeyPoint> keypoints);
    void set_unknown_keypoints(std::vector<cv::KeyPoint> keypoints);
    void set_thing_keypoints(std::map<int,std::vector<cv::KeyPoint>> &keypoints);
    void set_thing_descriptor(std::map<int,cv::Mat> &descriptor);
    void set_unknown_descriptor(cv::Mat descriptor);
    void set_stuff_descriptor(cv::Mat descriptor);
    void setPanoptic(std::vector<Panoptic_Object> objs); 
    void set_panoptic_masks(cv::Mat masks);

    ConstraintPoseImu* mpcpi;

    bool imuIsPreintegrated();
    void setIntegrated();

    bool isSet() const;

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline Eigen::Vector3f GetCameraCenter(){
        return mOw;
    }

    // Returns inverse of rotation
    inline Eigen::Matrix3f GetRotationInverse(){
        return mRwc;
    }

    inline Sophus::SE3<float> GetPose() const {
        //TODO: can the Frame pose be accsessed from several threads? should this be protected somehow?
        return mTcw;
    }

    inline Eigen::Matrix3f GetRwc() const {
        return mRwc;
    }

    inline Eigen::Vector3f GetOw() const {
        return mOw;
    }

    inline bool HasPose() const {
        return mbHasPose;
    }

    inline bool HasVelocity() const {
        return mbHasVelocity;
    }

    // Panoptic functions
    void SetPanopticResults(cv::Mat &people_mask);

    // RGB-D
    void RemoveOutliers(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera);
    void Remove_Moving_KeyPoints(std::vector<cv::KeyPoint> kps, const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera);

private:
    //Sophus/Eigen migration
    Sophus::SE3<float> mTcw;
    Eigen::Matrix<float,3,3> mRwc;
    Eigen::Matrix<float,3,1> mOw;
    Eigen::Matrix<float,3,3> mRcw;
    Eigen::Matrix<float,3,1> mtcw;
    bool mbHasPose;

    //Rcw_ not necessary as Sophus has a method for extracting the rotation matrix: Tcw_.rotationMatrix()
    //tcw_ not necessary as Sophus has a method for extracting the translation vector: Tcw_.translation()
    //Twc_ not necessary as Sophus has a method for easily computing the inverse pose: Tcw_.inverse()

    Sophus::SE3<float> mTlr, mTrl;
    Eigen::Matrix<float,3,3> mRlr;
    Eigen::Vector3f mtlr;


    // IMU linear velocity
    Eigen::Vector3f mVw;
    bool mbHasVelocity;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
    Eigen::Matrix3f mK_;
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float mThDepth;

    // Number of KeyPoints.
    int N;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
    std::vector<cv::KeyPoint> mvKeysUn;
    
    // Corresponding stereo coordinate and depth for each keypoint.
    std::vector<MapPoint*> mvpMapPoints;
    // "Monocular" keypoints have a negative value.
    std::vector<float> mvuRight;
    std::vector<float> mvDepth;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat mDescriptors, mDescriptorsRight;

    // MapPoints associated to keypoints, NULL pointer if no association.
    // Flag to identify outlier associations.
    std::vector<bool> mvbOutlier;
    int mnCloseMPs;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    IMU::Bias mPredBias;

    // IMU bias
    IMU::Bias mImuBias;

    // Imu calibration
    IMU::Calib mImuCalib;

    // Imu preintegration from last keyframe
    IMU::Preintegrated* mpImuPreintegrated;
    KeyFrame* mpLastKeyFrame;

    // Pointer to previous frame
    Frame* mpPrevFrame;
    IMU::Preintegrated* mpImuPreintegratedFrame;

    // Current and Next Frame id.
    static long unsigned int nNextId;
    long unsigned int mnId;

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;

    // Scale pyramid info.
    int mnScaleLevels;
    float mfScaleFactor;
    float mfLogScaleFactor;
    vector<float> mvScaleFactors;
    vector<float> mvInvScaleFactors;
    vector<float> mvLevelSigma2;
    vector<float> mvInvLevelSigma2;

    // Undistorted Image Bounds (computed once).
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

    static bool mbInitialComputations;

    map<long unsigned int, cv::Point2f> mmProjectPoints;
    map<long unsigned int, cv::Point2f> mmMatchedInImage;

    string mNameFile;

    int mnDataset;

#ifdef REGISTER_TIMES
    double mTimeORB_Ext;
    double mTimeStereoMatch;
#endif

private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    bool mbIsSet;

    bool mbImuPreintegrated;

    std::mutex *mpMutexImu;

    //Panoptic variables
    cv::Mat peopleMask;
    

public:
    GeometricCamera* mpCamera, *mpCamera2;

    //Number of KeyPoints extracted in the left and right images
    int Nleft, Nright;
    //Number of Non Lapping Keypoints
    int monoLeft, monoRight;

    //For stereo matching
    std::vector<int> mvLeftToRightMatch, mvRightToLeftMatch;

    //For stereo fisheye matching
    static cv::BFMatcher BFmatcher;

    //Triangulated stereo observations using as reference the left camera. These are
    //computed during ComputeStereoFishEyeMatches
    std::vector<Eigen::Vector3f> mvStereo3Dpoints;

    //Grid for the right image
    std::vector<std::size_t> mGridRight[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, Sophus::SE3f& Tlr,Frame* pPrevF = static_cast<Frame*>(NULL), const IMU::Calib &ImuCalib = IMU::Calib());

    //Stereo fisheye
    void ComputeStereoFishEyeMatches();

    bool isInFrustumChecks(MapPoint* pMP, float viewingCosLimit, bool bRight = false);

    Eigen::Vector3f UnprojectStereoFishEye(const int &i);

    cv::Mat imgLeft, imgRight;

    void PrintPointDistribution(){
        int left = 0, right = 0;
        int Nlim = (Nleft != -1) ? Nleft : N;
        for(int i = 0; i < N; i++){
            if(mvpMapPoints[i] && !mvbOutlier[i]){
                if(i < Nlim) left++;
                else right++;
            }
        }
        cout << "Point distribution in Frame: left-> " << left << " --- right-> " << right << endl;
    }

    Sophus::SE3<double> T_test;
};

}// namespace ORB_SLAM

#endif // FRAME_H
