/**
 * @file gather_surfaces.cpp
 * @author Nate Novak
 * @brief Program to gather data on surfaces to project the board onto. 
 * @date 2022-04-21
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include "../include/csv_util.h"
#include "../include/ar.h"

int main(int argc, char *argv[]) {
  cv::VideoCapture *capdev; // open the video device 
  capdev = new cv::VideoCapture(0);
  if( !capdev->isOpened() ) {
    printf("Unable to open video device\n");
    return(-1);
  }

  // get some properties of the image
  cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                  (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
  printf("Expected size: %d %d\n", refS.width, refS.height);
  
  cv::namedWindow("Gather Plane Data", 1); 
  cv::Mat frame;
  cv::Mat dst; 
  cv::Mat gray; 
  cv::Mat prev_image; 
  for(;;) {
    *capdev >> frame; // get a new frame from the camera, treat as a stream
    if( frame.empty() ) {
      printf("frame is empty\n");
      break;
    }  

    frame.copyTo(gray);
    std::vector<cv::Point2f> corner_set; // Image points of the corners of the chessboard
    std::vector<cv::Vec3f> point_set; // World coords
    cv::Mat rotations; // Rotational vector
    cv::Mat translations; // Translational vector
    std::vector<cv::Point2f> image_points; // Image points to project onto the scene
    
    cv::Ptr<cv::ORB> orb = cv::ORB::create(); 

    
    // Convert to grayscale
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); 
    std::vector<cv::KeyPoint> keypoints; 
    cv::Mat descriptors; 
    orb->detectAndCompute( gray, cv::noArray(), keypoints, descriptors );
    cv::drawKeypoints(gray, keypoints, dst); 
    cv::imshow("Gather Plane Data", dst);

    // check for quitting
    char keyEx = cv::waitKeyEx(10); 
    if(keyEx == 'q') {
      break; 
    } else if (keyEx == 's') {
      int id = std::rand() % 100; 
      std::string path = "./out_imgs/ex" + std::to_string(id) + ".png"; 
      cv::imwrite(path, dst); 
    } else if (keyEx == 'm') {
      printf("Yo\n"); 
      int id = std::rand() % 100; 
      std::string path = "./model_images/model" + std::to_string(id) + ".png";
      cv::imwrite(path, gray); 
    }
  }

  printf("Bye!\n"); 

  delete capdev;
  return 0; 
}