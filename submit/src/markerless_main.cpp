/**
 * @file gather_surfaces.cpp
 * @author Brenden Collins and Nate Novak
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
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "../include/csv_util.h"
#include "../include/markerless.h"

int main(int argc, char *argv[]) {
  cv::VideoCapture *capdev; // open the video device 
  capdev = new cv::VideoCapture(0);
  if( !capdev->isOpened() )
  {
    printf("Unable to open video device\n");
    return(-1);
  }

  bool drawkps = false; 
  if(argc > 1)
  {
    if(strcmp(argv[1], "-d") == 0)
    {
      printf("In Draw Keypoints Mode\n"); 
      drawkps = true; 
    } 
    else 
    {
      printf("error :: usage : use the flag -d to draw the matching keypoints\n"); 
      exit(-1); 
    }
  }

  // get some properties of the image
  cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
                  (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
  printf("Expected size: %d %d\n", refS.width, refS.height);

  // Delcare calibration data
  cv::Mat cam_mat(3, 3, CV_64FC1); 
  cv::Mat dist_coef(5, 1, CV_64FC1); 
  
  read_calibration_data_csv("calibration.csv", cam_mat, dist_coef, 0); 

  std::string winName= "Markerless AR"; 
  cv::namedWindow(winName, 1); 
  cv::Mat frame;
  cv::Mat dst; 
  cv::Mat gray; 

  cv::Ptr<cv::ORB> orb = cv::ORB::create();  // Create the ORB detector

  // Declare the keypoints and descriptors for the model images
  cv::Mat model; 
  std::vector<cv::KeyPoint> keypoints_model; 
  cv::Mat descriptors_model; 

  get_model_kp_desc(orb, model, keypoints_model, descriptors_model); 

  for(;;)
  {
    *capdev >> frame; // get a new frame from the camera, treat as a stream
    if( frame.empty() )
    {
      printf("frame is empty\n");
      break;
    }  

    // Convert to grayscale
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); 
    std::vector<cv::KeyPoint> keypoints_scene; 
    cv::Mat descriptors_scene; 
    orb->detectAndCompute( gray, cv::noArray(), keypoints_scene, descriptors_scene );

    // Match the keypoints
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);  
    std::vector<cv::DMatch> acceptable_matches; 
    bool sufficient_matches = false; 
    match_kps(matcher, descriptors_scene, descriptors_model, acceptable_matches, sufficient_matches); 
    frame.copyTo(dst); 
    
    if(drawkps)
    {
      cv::drawMatches(model, keypoints_model, gray, keypoints_scene, acceptable_matches, dst, 
                        cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
      cv::imshow(winName, dst); 
    }
    else
    {
      if(sufficient_matches)
      {
        cv::Mat rotations; 
        cv::Mat translations; 
        std::vector<cv::Point2f> scene_corners; 

        frame.copyTo(dst); 

        get_rots_and_trans(acceptable_matches, keypoints_model, keypoints_scene, model, rotations,
                           translations, cam_mat, dist_coef, scene_corners); 

        std::vector<cv::Vec3f> axespoints;  
        cv::Vec3f origin(0, 0, 0); 
        axes_points(axespoints, origin, 1.0);

        cv::Mat out_axes; 
        cv::projectPoints(axespoints, rotations, translations, cam_mat, dist_coef, out_axes); 

        cv::Point oo = cv::Point( out_axes.at<cv::Vec2f>(0,0) );
        cv::Point ox = cv::Point( out_axes.at<cv::Vec2f>(1,0) );
        cv::Point oy = cv::Point( out_axes.at<cv::Vec2f>(2,0) );
        cv::Point oz = cv::Point( out_axes.at<cv::Vec2f>(3,0) );
        cv::circle( dst, oo, 6, {255, 0, 0} );
        cv::circle( dst, oo, 8, {255, 0, 0} );
        cv::circle( dst, ox, 6, {255, 0, 255} );
        cv::circle( dst, ox, 8, {255, 0, 255} );
        cv::arrowedLine( dst, oo, ox, { 0, 0, 255 }, 2);
        cv::arrowedLine( dst, oo, oy, { 0, 255, 0 }, 2 );
        cv::arrowedLine( dst, oo, oz, { 255, 0, 0 }, 2 );

      } else {
        frame.copyTo(dst); 
      }
    }
    
    cv::imshow(winName, dst); 
    char keyEx = cv::waitKeyEx(10); 
    if(keyEx == 'q') {
      break; 
    } else if (keyEx == 's') {
      int id = std::rand() % 100; 
      std::string imgname; 
      printf("Please enter the name of the images.\n"); 
      std::cin >> imgname; 
      std::string path = "./out_imgs/" + imgname + ".png"; 
      cv::imwrite(path, dst); 
    }
  }

  printf("Bye!\n"); 

  delete capdev;
  return 0; 
}
