/**
 * @file linear_algebra.h
 * @author Nate Novak (novak.n@northeastern.edu)
 * @brief Function library for doing markerless AR
 * @date 2022-04-27
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

/**
 * @brief Function to find the model's keypoints and descriptors for those keypoints
 * 
 * @param detector ORB detector 
 * @param model input array for model image
 * @param keypoints input array keypoints found on the image.
 * @param descriptors descriptors for the keypoints found in the model. 
 */
void get_model_kp_desc(cv::Ptr<cv::ORB> detector, cv::Mat &model, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors); 

/**
 * @brief Function to match keypoints in the scene and the model
 * 
 * @param matcher DiscriptorMatcher that matches descriptors to the key points
 * @param desc_scene input array of descriptors of keypoints in the scene
 * @param desc_model input array of descriptors of keypoints in the model
 * @param acceptable_matches output vector of the top matches between the scene and model
 * @param enough boolean that indicates whether or not there are sufficient keypoint matches between the scene and boolean 
 */
void match_kps(cv::Ptr<cv::DescriptorMatcher> matcher, cv::Mat &desc_scene, cv::Mat &desc_model, std::vector<cv::DMatch> &acceptable_matches, bool &enough); 

/**
 * @brief Function to get the rotations and translations from the solvePnP opencv function
 * 
 * @param matches keypoint matches
 * @param keypoints_model keypoints in the model
 * @param keypoints_scene keypoints in the scene
 * @param model the model image
 * @param rotations output array of the rotations
 * @param translations output array of the translations
 * @param cam_mat camera matrix 
 * @param dist_coeffs distortion coefficients
 */
void get_rots_and_trans(std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &keypoints_model, std::vector<cv::KeyPoint> &keypoints_scene, 
                        cv::Mat &model, cv::Mat &rotations, cv::Mat &translations, cv::Mat cam_mat, cv::Mat dist_coeffs); 

/**
 * @brief Function to draw the axes
 * 
 * @param points vector of world points to write to
 * @param origin origin of the axes
 * @param scale scale of the axes
 * @param color color of the axes
 * @return int 
 */
int axes_points(std::vector<cv::Vec3f> &points, cv::Vec3f origin, float scale); 