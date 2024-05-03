/*
* This file is part of Panoptic-SLAM.
* Copyright (C) 2024 Gabriel Fischer Abati & João Carlos Virgolino Soares 
* Istituto Italiano di Tecnologia
* Pontifical Catholic University of Rio de Janeiro - PUC-Rio
* Co-authors: Vivian Suzano Medeiros, Marco Antonio Meggiolaro and Claudio Semini
* Please report suggestions and comments to gabriel.fischer@iit.it
*/

#include "PanopticNet.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <dirent.h>
#include <errno.h>
#include <experimental/filesystem>    // comment this for python 3.8
#include <unistd.h>
#include <memory>

#include <thread>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iostream>

// C PYTHON API REFERENCES
//http://books.gigatux.nl/mirror/pythonprogramming/0596000855_python2-CHP-20-SECT-5.html
//https://gist.github.com/Xonxt/26d2a9ac6c56505d0896822ede99a646?permalink_comment_id=3619671
//https://edcjones.tripod.com/refcount.html  Py_INCREF & Py_DECREF

void signal_callback_handler(int signum) {
   cout << "Terminate program \n";// << signum << endl;
   // Terminate program
   exit(signum);
}

namespace ORB_SLAM3
{
PanopticNet::PanopticNet(){

    mbNewImgFlag = false;
    mbFinishRequested = false;

    std::cout << "Importing Panoptic Segmentation Settings... \n";
    setup_env();
    Py_Initialize();

    PyEval_InitThreads();

    std::cout<<"Python Thread support Initialized \n";    
    _import_array();
        
    global_id = 0;
    next_tracking_id = 0;
    width = 640;
    height = 480;
    
    this->py_module = PyImport_ImportModule("panoptic_python"); //config_path.c_str()
    if (!py_check(this->py_module)) // check load py_Name 
    {
        py_error("python error: Cannot load module");
        //return;
	exit (EXIT_FAILURE);
    }
    
    this->py_class = PyObject_GetAttrString(this->py_module, (char*)"Panoptic_FPN_Net");
    if (!py_check(this->py_class)) // check load py_module 
    {
        py_error("python error: Cannot load class");
        //return;
        exit (EXIT_FAILURE);
    }
    
    //Check if PyObject is callable
    if(this->py_class && PyCallable_Check(this->py_class))
    {
        std::string config_path_tmp = "config/Panoptic_SLAM.yaml";
        PyObject *pargs  = Py_BuildValue("(s)", config_path_tmp.c_str());
        this->py_instance = PyEval_CallObject(this->py_class,pargs); //call class __init__
        if (!py_check(this->py_instance)) // check load py_instance
        {
            py_error("python error: Cannot load class");
            exit (EXIT_FAILURE); 
        }

        this->py_method =  PyObject_GetAttrString(this->py_instance, (char*)"panoptic_run_cpp"); 
        if (!py_check(this->py_method)) // check load py_method
        {
            py_error("python error: Cannot load method panoptic_run_cpp");
            exit (EXIT_FAILURE);
        }

        this->py_get_image = PyObject_GetAttrString(this->py_instance, (char*)"get_output_img");
        if (!py_check(this->py_get_image)) // check load py_method
        {
            py_error("python error: Cannot load method get_output_img");
            exit (EXIT_FAILURE);
        }

        this->py_get_instance_mask = PyObject_GetAttrString(this->py_instance, (char*)"get_all_instance_mask");
        if (!py_check(this->py_get_instance_mask)) // check load py_method
        {
            py_error("python error: Cannot load method get_all_instance_mask");
            exit (EXIT_FAILURE);
        }

        this->py_get_all_masks = PyObject_GetAttrString(this->py_instance, (char*)"get_all_masks");
        if (!py_check(this->py_get_all_masks)) // check load py_method
        {
            py_error("python error: Cannot load method get_all_masks");
            return;
        }

        std::cout<<"Panoptic Loaded \n";

    }
    else
    {
        py_error("python error: Module is not callable");
        exit (EXIT_FAILURE);
    }
//
    }

//=========================================================
void PanopticNet::setup_env()
{
    //pwd get current folder
    char buff[FILENAME_MAX]; //create string buffer to hold path
    if(getcwd( buff, FILENAME_MAX ) ==NULL)
    {
        std::cerr<<"Error: cannot get current working folder \n";
        return;
    }
    std::string pwd(buff);

    //setting automatic PYTHONPATH environment variable
    std::string python_path = "PYTHONPATH=" + pwd + "/src/python/";
    std::cout << python_path << "\n";
    char * py_path = new char[python_path.size() + 1];
    std::copy(python_path.begin(), python_path.end(), py_path);
    py_path[python_path.size()] = '\0'; // don't forget the terminating 0
    
    putenv(py_path);
}
//=========================================================
bool PanopticNet::py_check(PyObject *in)
{
    if (in == nullptr) {return false;}
    return true;
}
//==========================================================
void PanopticNet::py_error(std::string msg)
{
    PyErr_Print();
    std::cerr << msg << std::endl;
}
//=========================================================
void PanopticNet::PrintPyObject(PyObject *obj) // DEBUG ONLY
{
    PyObject* objectsRepresentation = PyObject_Repr(obj);
    const char* s = PyUnicode_AsUTF8(objectsRepresentation);
    std::cout<<s<<std::endl;
}
//==========================================================
void PanopticNet::PyNumpy_AsCV(PyObject* pData, int rows, int cols, cv::Mat &mask)
{
    uchar *data = (uchar *)PyByteArray_AsString(pData);
    cv::Mat img(rows, cols, CV_8UC3, data);
    mask = img.clone();
}
//==========================================================
cv::Rect PanopticNet::PyList_AsRect(PyObject* pData)
{
    cv::Rect result;
    std::vector<float> tmp;
    if(PyList_Check(pData)) {
        for(Py_ssize_t i = 0; i < PyList_Size(pData); i++) 
        {
            PyObject *value = PyList_GetItem(pData, i);
            Py_INCREF(value);
            tmp.push_back( PyFloat_AsDouble(value) );
            Py_DECREF(value);
        }
    }
    else {return cv::Rect(0,0,1,1);} // empty bounding box

    result.x = tmp[0];
    result.y = tmp[1];
    result.width = tmp[2];
    result.height = tmp[3];
			
    return result;
}
//==========================================================
cv::Mat PanopticNet::GetPanoptic_image()
{
    //Run Python panoptic method
    PyObject *pValue = PyEval_CallObject(this->py_get_image,NULL);
    cv::Mat img;
    PyNumpy_AsCV(pValue,480,640, img); 
    Py_DECREF(pValue);
    return img;
    
}
//==========================================================
cv::Mat PanopticNet::GetUnion_Instance_Mask()
{
    PyObject *pValue = PyEval_CallObject(this->py_get_instance_mask,NULL);
    cv::Mat img;
    PyNumpy_AsCV(pValue,480,640, img);
    Py_DECREF(pValue);
    return img;
}
//==========================================================
cv::Mat PanopticNet::GetAllMasks()
{
    PyObject *pValue = PyEval_CallObject(this->py_get_all_masks,NULL);
    cv::Mat img;
    PyNumpy_AsCV(pValue,480,640,img);
    Py_DECREF(pValue);
    return img;
}
//==========================================================
Panoptic_image PanopticNet::GetPanoptic()
{
    
    cv::Mat image = mImg;
    
    Panoptic_image results;
    std::vector<Panoptic_Object> objs;
    
    uchar *m = image.data;
    npy_intp mdim[3] = {image.rows,image.cols,image.channels()};  // the dimensions of the matrix
    
    const int ND = 3;
    
    PyObject* mat = PyArray_SimpleNewFromData(ND, mdim, NPY_UINT8, reinterpret_cast<void*>(m)); //convert the cv::Mat to numpy.array

    PyObject* py_args = Py_BuildValue("(O)", mat);

    //Run Python panoptic method
    PyObject *pValue = PyEval_CallObject(this->py_method,py_args);
    if(pValue == NULL) {std::cout<<"Error in python code \n";}

    //PyObject convertion variables
    int id=0;
    int isThing=0;
    float score=0.0;
    int category_id=0;
    int instance_id=0;  
    long area=0;  
    cv::Rect bbox;
    cv::Mat mask;

    //convert pValue to C
    if (PyList_Check(pValue)) 
    {
        
        for(Py_ssize_t i = 0; i < PyList_Size(pValue); i++) 
        {
            PyObject *value = PyList_GetItem(pValue, i);
            Py_INCREF(value);

            Panoptic_Object obj;

            for(Py_ssize_t j=0; j<PyList_Size(value); j++) //[id,isThing,score,category_id,instance_id,area,mask,bbox]
            {
                PyObject *sub_value = PyList_GetItem(value, j);
                Py_INCREF(sub_value);
                switch (j)
                {
                    case Element::id_elem:      {id = PyLong_AsLong(sub_value); break;}
                    case Element::isThing_elem: {isThing = PyLong_AsLong(sub_value); break;}
                    case Element::score_elem:    {score = PyFloat_AsDouble(sub_value); break;}
                    case Element::category_elem: {category_id = PyLong_AsLong(sub_value);break;}
                    case Element::instance_elem: {instance_id = PyLong_AsLong(sub_value); break;}
                    case Element::area_elem:     {area = PyLong_AsLong(sub_value); break;}
                    case Element::mask_elem:     {PyNumpy_AsCV(sub_value,480,640,mask); break;}
                    case Element::bbox_elem:     {bbox = PyList_AsRect(sub_value); break;}
                }
                Py_DECREF(sub_value); 
            }

            //consider cardboard as Thing object
            if(category_id==4 && !isThing){isThing=true;}

            //append object to object list
            obj.id = id;                   // not used 
            obj.isThing = isThing;
            obj.score = score;
            obj.category_id = category_id;
            obj.instance_id = instance_id; //not used
            obj.area = area;
            obj.mask = mask.clone();
            obj.bbox = bbox;
            objs.push_back(obj);

            Py_DECREF(value);
            
        }
       
    }
    else  {std::cout<<"Panoptic result is not a list"<<std::endl;}

    //get union from all instance masks
    results.union_instance_mask = GetUnion_Instance_Mask();
    //get output image 
    results.image = GetPanoptic_image();

    results.all_masks = GetAllMasks();

    //store objects in struct
    results.objs = objs;
    results.id = global_id; // define frame id
    global_id++; //update global frame id

    Py_DECREF(mat);
    Py_DECREF(py_args);
    Py_DECREF(pValue);
    
    return results;
}
//==========================================================
Panoptic_image PanopticNet::GetPanoptic(cv::Mat &image)
{

    Panoptic_image results;
    std::vector<Panoptic_Object> objs;
    
    uchar *m = image.data;
    npy_intp mdim[3] = {image.rows,image.cols,image.channels()};  // the dimensions of the matrix
    
    const int ND = 3;
    PyObject* mat = PyArray_SimpleNewFromData(ND, mdim, NPY_UINT8, reinterpret_cast<void*>(m)); //convert the cv::Mat to numpy.array
    
    
    PyObject* py_args = Py_BuildValue("(O)", mat);

    //Run Python panoptic method
    PyObject *pValue = PyEval_CallObject(this->py_method,py_args);
    if(pValue == NULL) {std::cout<<"Error in python code \n";}

    //PyObject convertion variables
    int id=0;
    int isThing=0;
    float score=0.0;
    int category_id=0;
    int instance_id=0;  
    long area=0;  
    cv::Rect bbox;
    cv::Mat mask;

    //convert pValue to C
    if (PyList_Check(pValue)) 
    {
        for(Py_ssize_t i = 0; i < PyList_Size(pValue); i++) 
        {
            PyObject *value = PyList_GetItem(pValue, i);
            Py_INCREF(value);

            Panoptic_Object obj;
            for(Py_ssize_t j=0; j<PyList_Size(value); j++) //[id,isThing,score,category_id,instance_id,area,mask,bbox]
            {
                PyObject *sub_value = PyList_GetItem(value, j);
                Py_INCREF(sub_value);
                switch (j)
                {
                    case Element::id_elem:      {id = PyLong_AsLong(sub_value); break;}
                    case Element::isThing_elem: {isThing = PyLong_AsLong(sub_value); break;}
                    case Element::score_elem:    {score = PyFloat_AsDouble(sub_value); break;}
                    case Element::category_elem: {category_id = PyLong_AsLong(sub_value);break;}
                    case Element::instance_elem: {instance_id = PyLong_AsLong(sub_value); break;}
                    case Element::area_elem:     {area = PyLong_AsLong(sub_value); break;}
                    case Element::mask_elem:     {PyNumpy_AsCV(sub_value,480,640,mask); break;}
                    case Element::bbox_elem:     {bbox = PyList_AsRect(sub_value); break;}
                } 
                Py_DECREF(sub_value); 
            }

            //append object to object list
            obj.id = id;
            obj.isThing = isThing;
            obj.score = score;
            obj.category_id = category_id;
            obj.instance_id = instance_id;
            obj.area = area;
            obj.mask = mask.clone();
            obj.bbox = bbox;
            objs.push_back(obj);

            Py_DECREF(value);
        }
    }
    else  {std::cout<<"Panoptic result is not a list"<<std::endl;}

    //get union from all instance masks
    results.union_instance_mask = GetUnion_Instance_Mask();

    //get output image 
    results.image = GetPanoptic_image();

    //store objects in struct
    results.objs = objs;
    results.id = global_id; // define frame id
    global_id++; //update global frame id

    Py_DECREF(mat);
    Py_DECREF(py_args);
    Py_DECREF(pValue);

    return results;
}
//=========================================================
std::vector<int> PanopticNet::Rect_2_vecInt(cv::Rect box)
{
    //convert bounding box cv::Rect format to int vector 
    int x1 = box.x;
    int y1 = box.y;
    int x2 = box.width;
    int y2 = box.height;

    std::vector<int> result = {x1,y1,x2,y2};
    return result;
}
//=========================================================
cv::Rect PanopticNet::convert_bbox(cv::Rect box)
{
    //convert bounding box cv::Rect format from [x1,y1,x2,y2] -> [x,y,w,h]
    int x1 = box.x;
    int y1 = box.y;
    int x2 = box.width;
    int y2 = box.height;

    int w = x2-x1+1;
    int h = y2-y1+1;

    cv::Rect result = cv::Rect(x1,y1,w,h);
    return result;
}
//========================================================
void PanopticNet::check_bbox(cv::Rect &bbox,int w, int h)
{
    //std::cout<<"Input x "<< bbox.x << " y "<< bbox.y <<" width "<< bbox.width << " height " << bbox.height << "\n";

    if((bbox.x + bbox.width) > w)   {bbox.width -= abs(w - bbox.x - bbox.width);}
    if((bbox.y + bbox.height) > h) {bbox.height -= abs(h - bbox.y - bbox.height);}

    //std::cout<<"Input x"<< bbox.x << " y "<< bbox.y <<" width "<< bbox.width << " height " << bbox.height << "\n";
}
//=========================================================
float PanopticNet::mask_iou(Panoptic_Object obj1, Panoptic_Object obj2)
{
    //check if objects are things
    if(!obj1.isThing || !obj2.isThing) {return 0.0;}

    //get object bounding box and change its format
    std::vector<int> bbox1 = Rect_2_vecInt(obj1.bbox);
    std::vector<int> bbox2 = Rect_2_vecInt(obj2.bbox);

    int x1 = std::max(bbox1[0],bbox2[0]);
    int y1 = std::max(bbox1[1],bbox2[1]);
    int x2 = std::min(bbox1[2],bbox2[2]);
    int y2 = std::min(bbox1[3],bbox2[3]);


    if(x1>x2 || y1>y2) {return 0.0;}

    int w = x2-x1+1;
    int h = y2-y1+1;

    //calculate area from mask 1
    cv::Rect mask1_bbox = convert_bbox(obj1.bbox);
    check_bbox(mask1_bbox,width,height); 
    cv::Mat mask1_crop = obj1.mask(mask1_bbox);
    
    cv::Mat gray_mask1;
    cv::cvtColor(mask1_crop,gray_mask1,cv::COLOR_BGR2GRAY);
    cv::threshold(gray_mask1,gray_mask1,1,1,cv::THRESH_BINARY);
    float area1 = sum(gray_mask1)[0];

    //get masks in the intersection part
    int start_ya = y1 - bbox1[1];
    int start_xa = x1 - bbox1[0];
    cv::Rect new_bbox1 = cv::Rect(start_xa,start_ya,w,h);
    check_bbox(new_bbox1,gray_mask1.cols,gray_mask1.rows);
    cv::Mat mask1 = gray_mask1(new_bbox1);
    //cv::imshow("tmp1",mask1);

    //calculate area from mask 2
    cv::Rect mask2_bbox = convert_bbox(obj2.bbox);
    check_bbox(mask2_bbox,width,height);
    cv::Mat mask2_crop = obj2.mask(mask2_bbox);
    cv::Mat gray_mask2;
    cv::cvtColor(mask2_crop,gray_mask2,cv::COLOR_BGR2GRAY);
    cv::threshold(gray_mask2,gray_mask2,1,1,cv::THRESH_BINARY);
    float area2 = sum(gray_mask2)[0];

    //get masks in the intersection part
    int start_yb = y1 - bbox2[1];
    int start_xb = x1 - bbox2[0];
    cv::Rect new_bbox2 = cv::Rect(start_xb,start_yb,w,h);
    check_bbox(new_bbox2,gray_mask2.cols,gray_mask2.rows);
    cv::Mat mask2 = gray_mask2(new_bbox2);
    //cv::imshow("tmp2",mask2);

    assert(mask1.size() == mask2.size());

    cv::Mat intersection;
    intersection = mask1 & mask2;

    float area_intersection = sum(intersection)[0];

    float area_union = area1 + area2 - area_intersection;

    float iou = area_intersection / area_union;

    return iou;
}
//=========================================================
void PanopticNet::ShortTerm_DA(std::vector<Panoptic_Object> &objs, std::vector<Panoptic_Object> &last_objs)
{
    /* Short Term Data Association */
    std::vector<std::vector<double>> iouMatrix;
    float iou_th = 0.15;
    iouMatrix.resize(objs.size(),std::vector<double>(last_objs.size(),0));

    for(unsigned int det=0; det< objs.size();det++)
    {	
        double iou;
        bool association_found = false;
        int lastObjlocalID;

        if(!objs[det].isThing)  {continue;}
        int current_category_id = objs[det].category_id;
        

        if(last_objs.size() != 0)
        {
            for(unsigned int lfo = 0; lfo<last_objs.size(); lfo++)
            {
                if(!last_objs[lfo].isThing) {continue;}
                int last_category_id = last_objs[lfo].category_id;

                iou = mask_iou(objs[det],last_objs[lfo]);
               
                if(isnan(iou)) {iouMatrix[det][lfo] = 0;}
                else {iouMatrix[det][lfo] = iou;}

                if(current_category_id == last_category_id)
                {   
                    if(iouMatrix[det][lfo] > iou_th)
                    {
                        association_found = true;
                        lastObjlocalID = lfo;
                        break;
                    }
                }

            }
        }
        
        if(association_found)
        {
            objs[det].tracking_id = last_objs[lastObjlocalID].tracking_id;
            objs[det].iou = iou;
            objs[det].isNewObj = false;
        }
        else
        {
            objs[det].tracking_id = next_tracking_id;
            objs[det].iou = iou;
            objs[det].isNewObj = true;
            next_tracking_id++;            
        }
    }
}
//=========================================================
PanopticNet::~PanopticNet(){
    delete this->py_module;
    delete this->py_class;
    delete this->py_instance;
    delete this->py_method;
    delete this->py_get_image;
    delete this->py_get_instance_mask;
    delete this->py_get_all_masks;

    Py_Finalize();
}

bool PanopticNet::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}
  
void PanopticNet::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested=true;
    Py_Finalize();
}

void PanopticNet::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void PanopticNet::Run()
{
    PyEval_SaveThread();
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();
    signal(SIGINT, signal_callback_handler);
    while(1)
    {		
	
        usleep(1);
        if(!isNewImgArrived()){
		continue;
	}
	

	
    Detect();
    

	if(isFinished())
    {
        PyGILState_Release(gstate);
        break;
    }

	
    }

}

void PanopticNet::Detect(){


    
    CurrentPanopticImg = GetPanoptic();
    
    pImg = CurrentPanopticImg.image;
    unkImg = CurrentPanopticImg.all_masks;

    SetDetectionFlag();		
    
}

bool PanopticNet::isNewImgArrived()
{
    unique_lock<mutex> lock(mMutexGetNewImg);
    if(mbNewImgFlag)
    {
        mbNewImgFlag=false;
        return true;
    }
    else
    	return false;
}


void PanopticNet::SetDetectionFlag()
{
    std::unique_lock <std::mutex> lock(mMutexNewImgDetection);
   
    mpTracker->mbNewDetImgFlag=true;
}


Panoptic_image PanopticNet::GetResults(){

    return CurrentPanopticImg;
}


}
