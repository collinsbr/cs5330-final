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
  
  std::string winName= "Matching"; 
  cv::namedWindow(winName, 1); 
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
    
    cv::Ptr<cv::ORB> orb = cv::ORB::create();  // Create the ORB detector
    
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
      while ((entry = readdir(dp))) {
        //printf("%s\n", entry->d_name); // uncomment to see what model images we have. 
        std::string name = entry->d_name; 
        if(name.compare(".") == 0 || name.compare("..") == 0) continue; 
        std::string path = dn_models + name; 
        cv::Mat model_raw = cv::imread(path, cv::IMREAD_GRAYSCALE);
        cv::resize(model_raw, model, cv::Size(640, 480)); 
        orb -> detectAndCompute( model, cv::noArray(), keypoints_model, descriptors_model); // get keypoints and descriptors from the model 
      }
    } else {
      printf("Could not find directory\n"); 
      exit(-1); 
    }

    // Match the keypoints
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);  
    std::vector< std::vector<cv::DMatch> > knn_matches; 
    if(descriptors_model.empty()) {
      printf("no descriptors in model\n"); 
      frame.copyTo(gray); 
      cv::imshow(winName, gray); 
      continue; 
    }
    if(descriptors_scene.empty()) {
      printf("no descriptors in scene\n"); 
      frame.copyTo(gray); 
      cv::imshow(winName, gray); continue; } 
    // Help from: https://stackoverflow.com/questions/29694490/flann-error-in-opencv-3
    // Convert to floats for the FLANN matcher
    if(descriptors_model.type() != CV_32F) {
      descriptors_model.convertTo(descriptors_model, CV_32F);
    }

    if(descriptors_scene.type() != CV_32F) {
      descriptors_scene.convertTo(descriptors_scene, CV_32F);
    }
    matcher->knnMatch(descriptors_model, descriptors_scene, knn_matches, 2);

    // Filter matches --> Lowe's ratio test
    const float thresh = 0.75;  
    std::vector<cv::DMatch> acceptable_matches; 
    for(int i = 0; i < knn_matches.size(); i++) {
      cv::DMatch cur_match_0 = knn_matches[i][0]; 
      cv::DMatch cur_match_1 = knn_matches[i][1]; 
      if(cur_match_0.distance < thresh * cur_match_1.distance) {
        acceptable_matches.push_back(cur_match_0); 
      }
    }

    if(acceptable_matches.size() < 15) {
      printf("Not sufficient matches\n"); 
      gray.copyTo(dst); 
      cv::imshow(winName, dst); 
      continue; 
    } 

    cv::drawMatches(model, keypoints_model, gray, keypoints_scene, acceptable_matches, dst, 
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS); 

    // localize object 
    std::vector<cv::Point2f> modelpts; 
    std::vector<cv::Point2f> scenepts; 
    // get keypoints from query index
    for(int i = 0; i < acceptable_matches.size(); i++) {
      modelpts.push_back(keypoints_model[acceptable_matches[i].queryIdx].pt); 
      scenepts.push_back(keypoints_scene[acceptable_matches[i].trainIdx].pt);
    }

    cv::Mat homography = cv::findHomography(modelpts, scenepts, cv::RANSAC); 

    // Get the corners from the model
    std::vector<cv::Point2f> model_corners(4); 
    model_corners[0] = cv::Point2f( 0, 0 ); 
    model_corners[1] = cv::Point2f( (float) model.cols, 0 ); 
    model_corners[2] = cv::Point2f( (float) model.cols, (float) model.rows ); 
    model_corners[3] = cv::Point2f( 0, float(model.rows) ); 

    
    cv::imshow(winName, dst); 
    // Now it's time to find the homography 
    // This section adds 
    //std::vector<cv::Point2f> src_points; 
    //std::vector<cv::Point2f> dst_points; 
    //for(int i = 0; i < best_matches.size(); i++) {
      //cv::DMatch match = best_matches[i]; 
      //src_points.push_back(keypoints_model[match.queryIdx].pt); 
      //dst_points.push_back(keypoints_scene[match.queryIdx].pt); 
    //}
    //cv::Mat homography = cv::findHomography(src_points, dst_points, cv::RANSAC, 5.0); // Testing homography 
    //int height= model.rows; 
    //int width = model.cols; 
    //std::vector<cv::Point2f> points {
      //cv::Point2f(0, 0), 
      //cv::Point2f(0, height - 1), 
      //cv::Point2f(width - 1, height - 1), 
      //cv::Point2f(width - 1, 0)
    //};

    /*
    //Project corners onto frame
    cv::Mat pt_out; 
    cv::perspectiveTransform(points, pt_out, homography); 
    printf("%d\n", pt_out.type()); 

    cv::Mat pt_out_int(pt_out.size(), CV_32S); 
    printf("%d %d\n", pt_out.size().width, pt_out.size().height); 
    printf("%.4f\n", pt_out.at<cv::Point2f>(0, 0).x); 
    printf("%.4f\n", pt_out.at<cv::Point2f>(0, 0).y); 
    for(int i = 0; i < pt_out.rows; i++) {
      for(int j = 0; j < pt_out.cols; j++) {
        pt_out_int.at<cv::Point2i>(i, j).x = (int) pt_out.at<cv::Point2f>(i, j).x; 
        pt_out_int.at<cv::Point2i>(i, j).y = (int) pt_out.at<cv::Point2f>(i, j).y; 
      }
    }

    cv::polylines(frame, pt_out, true, 255, 3); 
    */
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