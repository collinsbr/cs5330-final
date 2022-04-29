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
#include "../include/markerless.h"
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

  // Delcare calibration data
  cv::Mat cam_mat(3, 3, CV_64FC1); 
  cv::Mat dist_coef(5, 1, CV_64FC1); 
  
  read_calibration_data_csv("calibration.csv", cam_mat, dist_coef, 0); 

  std::string winName= "Matching"; 
  cv::namedWindow(winName, 1); 
  cv::Mat frame;
  cv::Mat dst; 
  cv::Mat gray; 

  
  char dirname[] = "./model_images/"; // directory for the model
  std::string dn_models = "./model_images/"; 
  struct dirent *entry = nullptr; 
  DIR *dp = nullptr;
  dp = opendir(dirname); 
  
  cv::Ptr<cv::ORB> orb = cv::ORB::create();  // Create the ORB detector

  cv::Mat model; 
  std::vector<cv::KeyPoint> keypoints_model; 
  cv::Mat descriptors_model; 

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
  for(;;) {
    *capdev >> frame; // get a new frame from the camera, treat as a stream
    if( frame.empty() ) {
      printf("frame is empty\n");
      break;
    }  

    frame.copyTo(gray);

    
    // Convert to grayscale
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); 
    std::vector<cv::KeyPoint> keypoints_scene; 
    cv::Mat descriptors_scene; 
    orb->detectAndCompute( gray, cv::noArray(), keypoints_scene, descriptors_scene );



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
      gray.copyTo(dst); 
      cv::imshow(winName, dst); 
      continue; 
    } 
   //VISUALIZATION OF KEYPOINTS
   //cv::drawMatches(model, keypoints_model, gray, keypoints_scene, acceptable_matches, dst, 
                    //cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    //cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // localize object 
    std::vector<cv::Point2f> modelpts; 
    std::vector<cv::Point2f> scenepts; 
    // get keypoints from query index
    for(int i = 0; i < acceptable_matches.size(); i++) {
      modelpts.push_back(keypoints_model[acceptable_matches[i].queryIdx].pt); 
      scenepts.push_back(keypoints_scene[acceptable_matches[i].trainIdx].pt);
    }
    frame.copyTo(dst); 
    cv::Mat rotations; 
    cv::Mat translations;
    std::vector<cv::Point2f> image_points;    

    cv::Mat homography = cv::findHomography(modelpts, scenepts, cv::RANSAC);
    // Get the corners from the model
    std::vector<cv::Point2f> model_corners(4); 
    model_corners[0] = cv::Point2f( 0, 0 ); 
    model_corners[1] = cv::Point2f( (float) model.cols, 0 ); 
    model_corners[2] = cv::Point2f( (float) model.cols, (float) model.rows ); 
    model_corners[3] = cv::Point2f( 0, float(model.rows) );
    std::vector<cv::Point2f> scene_corners(4); 
    cv::perspectiveTransform(model_corners, scene_corners, homography); 
    
    //cv::circle( dst, scene_corners[0], 6, {255, 255, 255}); 
    //cv::circle( dst, scene_corners[1], 6, {255, 0, 0}); 
    //cv::circle( dst, scene_corners[2], 6, {0, 255, 0}); 
    //cv::circle( dst, scene_corners[3], 6, {0, 0, 255}); 
    
    //std::vector<cv::Vec3f> point_set {
      //cv::Vec3f(0, 0, 0), 
      //cv::Vec3f(0, -1, 0),
      //cv::Vec3f(-1, -1, 0),
      //cv::Vec3f(-1, 0, 0)
    //}; 

    //cv::solvePnP(point_set, scene_corners, cam_mat, dist_coef, rotations, translations); 

    //std::vector<cv::Vec3f> axespoints;  
    //axespoints.push_back( cv::Vec3f(0, 0, 0) );
    //axespoints.push_back( cv::Vec3f(0.5, 0, 0) );
    //axespoints.push_back( cv::Vec3f(0, 0.5, 0) );
    //axespoints.push_back( cv::Vec3f(0, 0, 0.5) );
    
    //cv::Mat out_axes; 
    //cv::projectPoints( axespoints, rotations, translations, cam_mat, dist_coef, out_axes); 
    //cv::Point oo = cv::Point( out_axes.at<cv::Vec2f>(0,0) );
    //cv::Point ox = cv::Point( out_axes.at<cv::Vec2f>(1,0) );
    //cv::Point oy = cv::Point( out_axes.at<cv::Vec2f>(2,0) );
    //cv::Point oz = cv::Point( out_axes.at<cv::Vec2f>(3,0) );
    //cv::circle( dst, oo, 6, {255, 0, 0} );
    //cv::circle( dst, oo, 8, {255, 0, 0} );
    //cv::circle( dst, ox, 6, {255, 0, 255} );
    //cv::circle( dst, ox, 8, {255, 0, 255} );
    //cv::arrowedLine( dst, oo, ox, { 0, 0, 255 }, 2);
    //cv::arrowedLine( dst, oo, oy, { 0, 255, 0 }, 2 );
    //cv::arrowedLine( dst, oo, oz, { 255, 0, 0 }, 2 );

    ////Draw the lines betwen the corners (mapped object in the scene)
////    cv::line( dst, scene_corners[0],
////      scene_corners[1], cv::Scalar(0, 255, 0), 4 );
////    cv::line( dst, scene_corners[1],
////      scene_corners[2], cv::Scalar( 0, 255, 0), 4 );
////    cv::line( dst, scene_corners[2],
////      scene_corners[3], cv::Scalar( 0, 255, 0), 4 );
////    cv::line( dst, scene_corners[3],
////      scene_corners[0], cv::Scalar( 0, 255, 0), 4 );

    // Computer rotation along x and y axis, and the translations 
    // 1.) Invert homography 
    invert_values(homography); 
    printf("Homography found\n"); 
    
    // 2.) dot product of multicplicative inverse of the camera parameters and the homography
    cv::Mat inv_cam_mat = multiplicative_inverse(cam_mat); 
    cv::Mat rots_and_trans = matrix_multiplication(inv_cam_mat, homography); 

    printf("Inverted\n"); 

    // Get the columns of the rotoations and translations
    cv::Mat rot_x_raw = get_column(rots_and_trans, 0); 
    
    printf("rot_x_raw:\n"); 
    for(int i = 0; i < rot_x_raw.rows; i++) {
      for(int j = 0; j < rot_x_raw.cols; j++) {
        printf("%.4f ", rot_x_raw.at<double>(i, j)); 
      }
      printf("\n"); 
    }
    
    cv::Mat rot_y_raw = get_column(rots_and_trans, 1); 
    
    printf("rot_y_raw:\n"); 
    for(int i = 0; i < rot_y_raw.rows; i++) {
      for(int j = 0; j < rot_y_raw.cols; j++) {
        printf("%.4f ", rot_y_raw.at<double>(i, j)); 
      }
      printf("\n"); 
    }
    
    cv::Mat trans_raw = get_column(rots_and_trans, 2); 
    printf("Columns retrieved\n"); 
    // normalize the vectors
    float rot_x_l2 = l2_norm(rot_x_raw); 
    float rot_y_l2 = l2_norm(rot_y_raw); 
    float norm_factor = sqrt(rot_x_l2 * rot_y_l2); 

    cv::Mat rot_x_norm = normalize_vector(rot_x_raw, norm_factor); 

    printf("rot_x_norm:\n"); 
    for(int i = 0; i < rot_x_norm.rows; i++) {
      for(int j = 0; j < rot_x_norm.cols; j++) {
        printf("%.4f ", rot_x_norm.at<double>(i, j)); 
      }
      printf("\n"); 
    }

    cv::Mat rot_y_norm = normalize_vector(rot_y_raw, norm_factor); 
    cv::Mat trans_norm = normalize_vector(trans_raw, norm_factor); 
    printf("Rotations and translations normalized\n"); 
    // Computer orthonormal basis
    // C, P, and D are mathematical terms used in the blogs
    cv::Mat c = add_vectors(rot_x_norm, rot_y_norm); 
    
    printf("c:\n"); 
    for(int i = 0; i < c.rows; i++) {
      for(int j = 0; j < c.cols; j++) {
        printf("%.4f ", c.at<double>(i, j)); 
      }
      printf("\n"); 
    }

    printf("vectors added\n"); 
    cv::Mat p = cross_product(rot_x_norm, rot_y_norm); 
    
    printf("p:\n"); 
    for(int i = 0; i < p.rows; i++) {
      for(int j = 0; j < p.cols; j++) {
        printf("%.4f ", p.at<double>(i, j)); 
      }
      printf("\n"); 
    }

    cv::Mat d = cross_product(c, p);

    printf("d:\n"); 
    for(int i = 0; i < d.rows; i++) {
      for(int j = 0; j < d.cols; j++) {
        printf("%.4f ", d.at<double>(i, j)); 
      }
      printf("\n"); 
    }

    printf("cross products of p and d\n"); 

    float c_l2 = l2_norm(c); 
    float d_l2 = l2_norm(d); 
    cv::Mat c_norm = normalize_vector(c, c_l2); // Didnt' need to write those functions evidently
    cv::Mat d_norm = normalize_vector(d, d_l2); 
    cv::Mat c_d_union = add_vectors(c_norm, d_norm); 

    printf("cnorm:\n"); 
    for(int i = 0; i < c_norm.rows; i++) {
      for(int j = 0; j < c_norm.cols; j++) {
        printf("%.4f ", c_norm.at<double>(i, j)); 
      }
      printf("\n"); 
    }

    cv::Mat c_d_intersection = subtract_vectors(c_norm, d_norm); 

    printf("dnorm:\n"); 
    for(int i = 0; i < d_norm.rows; i++) {
      for(int j = 0; j < d_norm.cols; j++) {
        printf("%.4f ", d_norm.at<double>(i, j)); 
      }
      printf("\n"); 
    }

    printf("c and d math done\n"); 

    const float multiplier = 1.0 / sqrt(2.0);
    
    cv::Mat rot_x_final = normalize_vector(c_d_union, multiplier); 
    cv::Mat rot_y_final = normalize_vector(c_d_intersection, multiplier);
    cv::Mat rot_z_final = cross_product(rot_x_final, rot_y_final);

    printf("x rots:\n"); 
    for(int i = 0; i < rot_x_final.rows; i++) {
      for(int j = 0; j < rot_x_final.cols; j++) {
        printf("%.4f ", rot_x_final.at<double>(i, j)); 
      }
      printf("\n"); 
    }

    printf("y rots:\n"); 
    for(int i = 0; i < rot_y_final.rows; i++) {
      for(int j = 0; j < rot_y_final.cols; j++) {
        printf("%.4f ", rot_y_final.at<double>(i, j)); 
      }
      printf("\n"); 
    }
 
    printf("z rots:\n"); 
    for(int i = 0; i < rot_z_final.rows; i++) {
      for(int j = 0; j < rot_z_final.cols; j++) {
        printf("%.4f ", rot_z_final.at<double>(i, j)); 
      }
      printf("\n"); 
    }

    printf("Final rotations found\n"); 

    std::vector<cv::Mat> matricies {
      rot_x_final, 
      rot_y_final,
      rot_z_final, 
      trans_norm
    }; 

   // for(int i= 0; i < matricies.size(); i++) {
   //   for(int j = 0; j < matricies[i].cols; j++){
   //     printf("%.4f ", matricies[i].at<double>(0, j)); 
   //   }
   //   printf("\n"); 
   // }

    cv::Mat stacked = stack(matricies); 
    cv::Mat stacked_transposed; 
    cv::transpose(stacked, stacked_transposed); 

    //for(int i = 0; i < stacked.rows; i++) {
    //  for(int j = 0; j < stacked.cols; j++) {
    //    printf("%.4f ", stacked.at<double>(i, j)); 
    //  }
    //  printf("\n"); 
    //}

    printf("Cam Mat -- w: %d h: %d\n", cam_mat.size().width, cam_mat.size().height); 
    printf("stacked -- w: %d h: %d\n", stacked.size().width, stacked.size().height); 
    printf("Stacked_transpose -- w: %d h: %d\n", stacked_transposed.size().width, stacked_transposed.size().height); 
    cv::Mat projection = cam_mat * stacked_transposed; 

    printf("projection determined\n"); 
    
    cv::imshow(winName, dst); 
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