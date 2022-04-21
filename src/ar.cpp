/**
 * @file ar.cpp
 * @author Nate Novak (novak.n@northeastern.edu)
 * @brief Library of functions for conducting augmented reality
 * @date 2022-03-23
 */ 
#include "../include/ar.h"

/**
 * @brief Function to detect and extract chessboard
 * 
 * @param src source image to find the corners in 
 * @param dst dst image that displays the the corners
 * @param patsize size of the pattern
 * @param corner_set vector of the point location of each corner  
 * @param pattern_found bool passed by reference to determine if corners were found. 
 * @return int return non-zero value on failure. 
 */
int detect_chessboard(const cv::Mat &src, cv::Size patsize, std::vector<cv::Point2f> &corner_set, bool &pattern_found) { 
  pattern_found = cv::findChessboardCorners(src, patsize, corner_set, cv::CALIB_CB_FAST_CHECK); 

  if(pattern_found) {
    cv::Mat gray; 
    cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY); 
    cv::cornerSubPix(gray, corner_set, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
  }   
  return 0; 
} 

/**
 * @brief Function to get the point set if there's a pattern
 * 
 * @param pat_size size of the pattern
 * @param point_set point set to write to
 * @return int N/A
 */
int get_point_set(const cv::Size pat_size, std::vector<cv::Vec3f> &point_set) {
  
  for(int i = 0; i < pat_size.height; i++) {
    for(int j = 0; j < pat_size.width; j++) {
      cv::Vec3f point3d; 
      point3d[0] = j; 
      point3d[1] = -i; 
      point3d[2] = 0; 
      point_set.push_back( point3d ); 
    }
  }

  return 0; 
}

/**
 * @brief Function to draw a cube 
 * 
 * @param points vector of world points to write to
 * @param origin origin of the cube
 * @param scale scale of the cube
 * @param color color of the cube
 * @return int 
 */
int draw_axes(std::vector<cv::Vec3f> &points, cv::Vec3f origin, float scale){
  points.push_back(cv::Vec3f(0, 0, 0)); // origin 
  points.push_back(cv::Vec3f(0, 0, 1)); // z axis
  points.push_back(cv::Vec3f(1, 0, 0)); // x axis
  points.push_back(cv::Vec3f(0, 1, 0)); // y axis

  return 0; 
} 