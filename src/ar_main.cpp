/**
 * @file ar_main.cpp
 * @author Nate Novak (novak.n@northeastern.edu)
 * @brief Main function for the AR portion of project 4
 * @date 2022-03-22
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
  cv::VideoCapture *capdev;

  // open the video device
  capdev = new cv::VideoCapture(0);
  if( !capdev->isOpened() ) {
    printf("Unable to open video device\n");
    return(-1);
  }

  // get some properties of the image
  cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                  (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
  printf("Expected size: %d %d\n", refS.width, refS.height);
  
  cv::namedWindow("Cal/AR", 1); 
  cv::Mat frame;
  cv::Mat dst; 

  // declare calibration data
  cv::Mat cam_mat(3, 3, CV_64FC1); 
  cv::Mat distcoeff(5, 1, CV_64FC1); 

  read_calibration_data_csv("calibration.csv", cam_mat, distcoeff, 0); 

  // Print the camera matrix
  printf("Camera Matrix\n"); 
  for(int i = 0; i < cam_mat.rows; i++) {
    for(int j = 0; j < cam_mat.cols; j++) {
      printf("%.4f ", cam_mat.at<double>(i, j)); 
    }
    printf("\n"); 
  }
  printf("\n"); 
  

  // Print the Distortion coeficients of the camera
  printf("Distortion Coefficients\n"); 
  for(int i = 0; i < distcoeff.rows; i++) {
    printf("%.4f ", distcoeff.at<double>(i, 0)); 
  }
  printf("\n\n"); 

  // Declare the size of the pattern
  cv::Size patternsize(9, 6); 

  // Flags to display various virtual objects I created. 
  bool show_vo = false;
  bool show_ext = false;  
  
  for(;;) {
    *capdev >> frame; // get a new frame from the camera, treat as a stream
    if( frame.empty() ) {
      printf("frame is empty\n");
      break;
    }  

    bool patternfound = false;

    std::vector<cv::Point2f> corner_set; // Image points of the corners of the chessboard
    std::vector<cv::Vec3f> point_set; // World coords
    cv::Mat rotations; // Rotational vector
    cv::Mat translations; // Translational vector
    std::vector<cv::Point2f> image_points; // Image points to project onto the scene

    detect_chessboard(frame, patternsize, corner_set, patternfound); 

    frame.copyTo(dst); 

    if(patternfound) {
      printf("pattern found\n"); 
      get_point_set(patternsize, point_set); // Get the point set for the panner
      cv::solvePnP(point_set, corner_set, cam_mat, distcoeff, rotations, translations);

      std::vector<cv::Vec3f> drawpoints;
      draw_axes(drawpoints, cv::Vec3f(0, 0, 0), 1);
      
      // project the points and get the image points  
      cv::projectPoints(drawpoints, rotations, translations, cam_mat, distcoeff, image_points);  
      
      // Draw the lines between the points
      cv::arrowedLine(dst, image_points[0], image_points[1], {255, 0, 0}, 2); // z
      cv::arrowedLine(dst, image_points[0], image_points[2], {0, 255, 0}, 2); // y
      cv::arrowedLine(dst, image_points[0], image_points[3], {0, 0, 255}, 2); // x
    }

    cv::imshow("Cal/AR", dst);

    // check for quitting
    char keyEx = cv::waitKeyEx(10); 
    if(keyEx == 'q') {
      break; 
    }
  }

  printf("Bye!\n"); 

  delete capdev;
  return(0);
}