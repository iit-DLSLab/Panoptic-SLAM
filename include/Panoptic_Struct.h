/*
* This file is part of Panoptic-SLAM.
* Copyright (C) 2024 Gabriel Fischer Abati & João Carlos Virgolino Soares 
* Istituto Italiano di Tecnologia
* Pontifical Catholic University of Rio de Janeiro - PUC-Rio
* Co-authors: Vivian Suzano Medeiros, Marco Antonio Meggiolaro and Claudio Semini
* Please report suggestions and comments to gabriel.fischer@iit.it
*/

#ifndef __PANOPTICSTRUCT_H
#define __PANOPTICSTRUCT_H

#include <opencv2/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>

namespace ORB_SLAM3
{
    struct Panoptic_Object
    {
        int id;         //not used
        cv::Mat mask; 
        bool isThing; //instance segmentation or semantic segmentation 
        float score; // confidence score for instance segmentation
        int category_id; // label id
        int instance_id; //not used
        float area; // mask area
        cv::Rect bbox; //bounding box
        unsigned int tracking_id; // short tearm data association id
        float iou;
        bool isNewObj;
    };

    struct Panoptic_image
    {
        unsigned int id;
        cv::Mat image;
        cv::Mat union_instance_mask;
        cv::Mat all_masks;
        std::vector<Panoptic_Object> objs;
    };
}

#endif
