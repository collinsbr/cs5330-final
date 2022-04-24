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
#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "../include/csv_util.h"

bool compareMatch(cv::DMatch dmatch1, cv::DMatch dmatch2) {
  return ( dmatch1.distance < dmatch2.distance ); 
}

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
  for(;;) {
    *capdev >> frame; // get a new frame from the camera, treat as a stream
    if( frame.empty() ) {
      printf("frame is empty\n");
      break;
    }  

    frame.copyTo(gray);

    //std::vector<cv::Point2f> corner_set; // Image points of the corners of the chessboard
    //std::vector<cv::Vec3f> point_set; // World coords
    //cv::Mat rotations; // Rotational vector
    //cv::Mat translations; // Translational vector
    //std::vector<cv::Point2f> image_points; // Image points to project onto the scene
    
    cv::Ptr<cv::ORB> orb = cv::ORB::create(); // Create the ORB detector
    
    // Convert to grayscale
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); 
    std::vector<cv::KeyPoint> keypoints_scene; 
    cv::Mat descriptors_scene; 
    orb->detectAndCompute( gray, cv::noArray(), keypoints_scene, descriptors_scene );

    // Read in the models and find the best match
    // This should change to read from a CSV file the keypoints
    // However, I am currently not confident in the structure of what the keypoints and descriptors look like
    // So I'll do that after prototyping
    std::vector<cv::KeyPoint> keypoints_model; 
    cv::Mat descriptors_model; 
    
    struct dirent *entry = nullptr; 
    DIR *dp = nullptr;
    
    char dirname[] = "./model_images/"; 
    std::string dn_models = "./model_images/"; 
    dp = opendir(dirname); 
    cv::Mat model; 

    if(dp != nullptr) {
      // Right now, just one model image.
      // Kept the for loop because at some point we will add more surfaces.
      while ((entry = readdir(dp))) {
        //printf("%s\n", entry->d_name); // uncomment to see what model images we have. 
        std::string name = entry->d_name; 
        if(name.compare(".") == 0 || name.compare("..") == 0) continue; 
        std::string path = dn_models + name; 
        model = cv::imread(path, cv::IMREAD_GRAYSCALE);
        orb -> detectAndCompute( model, cv::noArray(), keypoints_model, descriptors_model); // get keypoints and descriptors from the model 
      }
    } else {
      printf("Could not find directory\n"); 
      exit(-1); 
    }

    // Match the keypoints
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING); 
    std::vector<cv::DMatch> matches; 
    matcher->match(descriptors_scene, descriptors_model, matches);

    // sort matches 
    std::sort(matches.begin(), matches.end(), compareMatch);
    std::vector<cv::DMatch> best_matches; 
    // We'll probably want some threshold so if the best match is some distance away, we don't do anything  
    // Get the top 15 matches
    for(int i = 0; i < 15; i++) best_matches.push_back(matches[i]); 
   
    cv::drawMatches(gray, keypoints_scene, model, keypoints_model, best_matches, dst);
    cv::imshow("Gather Plane Data", dst);

    // check for quitting
    char keyEx = cv::waitKeyEx(10); 
    if(keyEx == 'q') {
      break; 
    } else if (keyEx == 's') {
      int id = std::rand() % 100; 
      std::string path = "./out_imgs/ex" + std::to_string(id) + ".png"; 
      cv::imwrite(path, dst); 
    }
  }

  printf("Bye!\n"); 

  delete capdev;
  return 0; 
}