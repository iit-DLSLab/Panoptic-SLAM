/*
* This file is part of Panoptic-SLAM.
* Copyright (C) 2024 Gabriel Fischer Abati & João Carlos Virgolino Soares 
* Istituto Italiano di Tecnologia
* Pontifical Catholic University of Rio de Janeiro - PUC-Rio
* Co-authors: Vivian Suzano Medeiros, Marco Antonio Meggiolaro and Claudio Semini
* Please report suggestions and comments to gabriel.fischer@iit.it
*/

#ifndef __PANOPTICNET_H
#define __PANOPTICNET_H

#ifndef NULL
#define NULL   ((void *) 0)
#endif


#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


#include <Python.h>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <cstdio>
#include "numpy/arrayobject.h"
//#include <boost/thread.hpp>
//#include "Conversion.h"

#include <opencv2/imgproc.hpp>
#include "Tracking.h"
#include <mutex>
#include <signal.h>
#include "Panoptic_Struct.h"

namespace ORB_SLAM3
{

class Tracking;

class PanopticNet{
private:
    PyObject *py_module; 	/*!< Module of python where the Mask algorithm is implemented */
    PyObject *py_class; 	/*!< Class to be instanced */
    PyObject *py_instance;  /*!< Instance of the class */
    PyObject *py_method; 	/*!< Panoptic method */		
    PyObject *py_get_image; /*!< get output Panoptic image */	
    PyObject *py_get_instance_mask; /*! get union of instance masks */	
    PyObject *py_get_all_masks; /* get all masks together*/
    
    std::string py_path; 	/*!< Path to be included to the environment variable PYTHONPATH */
    std::string module_name; /*!< Detailed description after the member */
    std::string class_name; /*!< Detailed description after the member */
    std::string get_dyn_seg; 	/*!< Detailed description after the member */

    unsigned long global_id;
    unsigned long next_tracking_id;
    int width,height; // image dimensions
    
    void py_error(std::string msg);
    bool py_check(PyObject *in);
    void PrintPyObject(PyObject *obj);
    void setup_env();
    void PyNumpy_AsCV(PyObject* pData, int rows, int cols, cv::Mat &mask);
    cv::Rect PyList_AsRect(PyObject* pData);
    cv::Mat GetPanoptic_image();
    cv::Mat GetUnion_Instance_Mask();
    cv::Mat GetAllMasks();
    std::vector<int> Rect_2_vecInt(cv::Rect box);
    cv::Rect convert_bbox(cv::Rect box);
    void check_bbox(cv::Rect &bbox, int w,int h);
    float mask_iou(Panoptic_Object obj1, Panoptic_Object obj2);

    enum Element {id_elem,isThing_elem,score_elem,category_elem,instance_elem,area_elem,mask_elem,bbox_elem};

    Panoptic_image CurrentPanopticImg;


public:

    PanopticNet();
    ~PanopticNet();

    Tracking *mpTracker;

    void Run();
    void SetTracker(Tracking *pTracker);
    bool isNewImgArrived();
    void SetDetectionFlag();
    void Detect();

    bool isFinished();
    void RequestFinish();

    cv::Mat mImg;
    cv::Mat pImg;	
    cv::Mat unkImg;
    std::mutex mMutexNewImgDetection;
    std::mutex mMutexGetNewImg;
    bool mbNewImgFlag;

    bool mbFinishRequested;
    std::mutex mMutexFinish;

    Panoptic_image GetPanoptic();
    Panoptic_image GetPanoptic(cv::Mat &image);
    Panoptic_image GetResults();
    void ShortTerm_DA(std::vector<Panoptic_Object> &objs, std::vector<Panoptic_Object> &last_objs);

    
};


}

#endif
