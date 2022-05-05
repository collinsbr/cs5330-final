/**
 * @file markerless.cpp
 * @author Nate Novak (novak.n@northeastern.edu)
 * @brief Function library of linear algebra functions
 * @date 2022-04-27
 */

#include "../include/markerless.h" 

/**
 * @brief Function to find the model's keypoints and descriptors for those keypoints
 * 
 * @param detector ORB detector 
 * @param model output array for model image
 * @param keypoints output array keypoints found on the image.
 * @param descriptors descriptors for the keypoints found in the model. 
 */
void get_model_kp_desc(cv::Ptr<cv::ORB> detector, cv::Mat &model, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) {
  // Set up the variables to read the directory to find the reference image 
  char dirname[] = "../model_images/"; // directory for the model
  std::string dn_models = "../model_images/"; 
  struct dirent *entry = nullptr; 
  DIR *dp = nullptr;
  dp = opendir(dirname); 
  
  if(dp != nullptr) {
    while ((entry = readdir(dp))) {
      std::string name = entry->d_name; 
      if(name.compare(".") == 0 || name.compare("..") == 0) continue; 
      if(name[0] == '.') continue;
      std::string path = dn_models + name; 
      std::cout << "pathname: " << path << std::endl;
      cv::Mat model_raw = cv::imread(path, cv::IMREAD_GRAYSCALE);
      printf("model_raw width: %d, height: %d\n", model_raw.size().width, model_raw.size().height);
      cv::resize(model_raw, model, cv::Size(640, 480)); 
      detector -> detectAndCompute( model, cv::noArray(), keypoints, descriptors); // get keypoints and descriptors from the model 
    }
  } else {
    printf("Could not find directory\n"); 
    exit(-1); 
  }
}

/**
 * @brief Function to match keypoints in the scene and the model
 * 
 * @param matcher DiscripterMatcher that matches descriptors to the key points
 * @param desc_scene input array of descriptors of keypoints in the scene
 * @param desc_model input array of descriptors of keypoints in the model
 * @param acceptable_matches output vector of the top matches between the scene and model
 * @param enough boolean that indicates whether or not there are sufficient keypoint matches between the scene and boolean 
 */
void match_kps(cv::Ptr<cv::DescriptorMatcher> matcher, cv::Mat &desc_scene, cv::Mat &desc_model, std::vector<cv::DMatch> &acceptable_matches, bool &enough) {
  // Check if there are any descriptors in the model
  if(desc_model.empty()) {
    printf("no descriptors in model\n"); 
    enough = false; 
    return; 
  }

  if(desc_scene.empty()) {
    printf("no descriptors in scene\n"); 
    enough = false; 
    return; 
  } 

  // Help from: https://stackoverflow.com/questions/29694490/flann-error-in-opencv-3
  // Convert to floats for the FLANN matcher
  if(desc_model.type() != CV_32F) {
    desc_model.convertTo(desc_model, CV_32F);
  }

  if(desc_scene.type() != CV_32F) {
    desc_scene.convertTo(desc_scene, CV_32F);
  }

  std::vector< std::vector<cv::DMatch> > knn_matches; 

  printf("model: %d,%d, scene: %d,%d\n", desc_model.size().width, desc_model.size().height,
                                         desc_scene.size().width, desc_scene.size().height );

  matcher->knnMatch(desc_model, desc_scene, knn_matches, 2); // Match the keypoints

  // Filter matches --> Lowe's ratio test
  const float thresh = 0.75;  
  for(int i = 0; i < knn_matches.size(); i++) {
    cv::DMatch cur_match_0 = knn_matches[i][0]; 
    cv::DMatch cur_match_1 = knn_matches[i][1]; 
    if(cur_match_0.distance < thresh * cur_match_1.distance) {
      acceptable_matches.push_back(cur_match_0); 
    }
  }

  if(acceptable_matches.size() < 18) {
    enough = false; 
    return; 
  } 

  enough = true; 
}

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
 * @param scene_corners output array of the corners of the surface in the scene
 */
void get_rots_and_trans(std::vector<cv::DMatch> &matches, std::vector<cv::KeyPoint> &keypoints_model,
       std::vector<cv::KeyPoint> &keypoints_scene, cv::Mat &model, cv::Mat &rotations, 
       cv::Mat &translations, cv::Mat cam_mat, std::vector<float>dist_coeffs, 
       std::vector<cv::Point2f> &scene_corners) {
  
  std::vector<cv::Point2f> modelpts; 
  std::vector<cv::Point2f> scenepts; 

  // get keypoints from query index
  printf("matches size: %lu\n", matches.size() );
  for(int i = 0; i < matches.size(); i++) {
    modelpts.push_back(keypoints_model[matches[i].queryIdx].pt); 
    scenepts.push_back(keypoints_scene[matches[i].trainIdx].pt);
  }

  cv::Mat homography = cv::findHomography(modelpts, scenepts, cv::LMEDS);
  // Get the corners from the model
  std::vector<cv::Point2f> model_corners(4); 
  model_corners[0] = cv::Point2f( 0, 0 ); 
  model_corners[1] = cv::Point2f( (float) model.cols, 0 ); 
  model_corners[2] = cv::Point2f( (float) model.cols, (float) model.rows ); 
  model_corners[3] = cv::Point2f( 0, float(model.rows) );

  // TODO: remove me
  printf("model size: %lu, scene size: %lu\n", model_corners.size(), scene_corners.size() );
  printf("homography size: %d, %d\n", homography.size().width, homography.size().height);
  
  cv::perspectiveTransform( model_corners, scene_corners, homography ); 

  std::vector<cv::Vec3f> point_set {
    cv::Vec3f(0, 0, 0),
    cv::Vec3f(0, -1, 0),
    cv::Vec3f(-1, -1, 0),
    cv::Vec3f(-1, 0, 0)
  }; 

  cv::solvePnP(point_set, scene_corners, cam_mat, dist_coeffs, rotations, translations); 

}

/**
 * @brief Function to draw the axes
 * 
 * @param points vector of world points to write to
 * @param origin origin of the axes
 * @param scale scale of the axes
 * @param color color of the axes
 * @return int 
 */
int axes_points(std::vector<cv::Vec3f> &points, cv::Vec3f origin, float scale) {
  points.push_back( cv::Vec3f(0, 0, 0) );
  points.push_back( cv::Vec3f(0.5, 0, 0) );
  points.push_back( cv::Vec3f(0, 0.5, 0) );
  points.push_back( cv::Vec3f(0, 0, 0.5) );
  return 0; 
}
