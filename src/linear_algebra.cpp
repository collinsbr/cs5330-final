/**
 * @file linear_algebra.h
 * @author Nate Novak (novak.n@northeastern.edu)
 * @brief Function library of linear algebra functions
 * @date 2022-04-27
 */

#include "../include/linear_algebra.h"

/**
 * @brief Function to multiply each value in a matrix by 01
 * 
 * @param matrix is the matrix to be multiplied by -1
 */
void invert_values(cv::Mat &matrix) {
  for(int i = 0; i < matrix.rows; i++) {
    for(int j = 0; j < matrix.cols; j++) {
      matrix.at<float>(i, j) = -1 * matrix.at<float>(i, j); 
    }
  }
}

/**
 * @brief Function to multiply two matrices
 * 
 * @param mat1 One of the matrices to multiply
 * @param mat2 The other matrix to multiply 
 * @return cv::Mat result of the mulitplication 
 */
cv::Mat matrix_multiplication(cv::Mat &mat1, cv::Mat &mat2) {
  cv::Mat out = mat1 * mat2;

  return out; 
}

/**
 * @brief Function to calculate the multiplicative inverse of a matrix
 * 
 * @param matrix matrix to be inverted
 * @return cv::Mat result of inversion
 */
cv::Mat multiplicative_inverse(cv::Mat &matrix) {
  cv::Mat out = matrix.inv();

  return out; 
}

/**
 * @brief Get the column of a specific matrix as a 1D Mat
 * 
 * @param matrix matrix from which to retrieve the column
 * @param colnum the column number
 * @return cv::Mat the output
 */
cv::Mat get_column(const cv::Mat &matrix, const int colnum) {
  cv::Mat out; 

  for(int i = 0; i < matrix.rows; i++) {
    out.at<float>(0, i) = matrix.at<float>(i, colnum); 
  }

  return out; 
}

/**
 * @brief Function to calculate the L2 norm of the vectors 
 * 
 * @param vector vector to determine the L2 norm of 
 * @return float result of the L2 norm calculation
 */
float l2_norm(const cv::Mat vector) {
  float sum = 0.0; 

  for(int i = 0; i < vector.cols; i++) {
    float value = vector.at<float>(0, i); 
    sum += value * value;
  }

  return sqrt(sum); 

}

/**
 * @brief Function to normalize a vector based on a normalization factor
 * 
 * @param vector vector to be normalized
 * @param norm_factor factor by which to normalize a vector
 * @return cv::Mat vector that holds normalized values
 */
cv::Mat normalize_vector(const cv::Mat &vector, const float norm_factor) {
  cv::Mat out; 
  for(int i = 0; i < vector.cols; i++) {
    out.at<float>(0, i) = ( vector.at<float>(0, i) / norm_factor ); 
  }

  return out; 
}

/**
 * @brief Function do an element-wise addition of two vectors
 * 
 * @param vec1 one vector to be added
 * @param vec2 other vector to be added
 * @return cv::Mat result of element-wise addition
 */
cv::Mat add_vectors(const cv::Mat &vec1, const cv::Mat &vec2) {
  cv::Mat out; 
  for(int i = 0; i < vec1.cols; i++) {
    float val = vec1.at<float>(0, i) + vec2.at<float>(0, i); 
    out.at<float>(0, i) = val; 
  }

  return out; 
}

/**
 * @brief Function to calculate the cross product of 2 3X1 vectors
 * 
 * @param vec1 one vectors of the cross product
 * @param vec2 the other vector of the cross product
 * @return cv::Mat the output vector of the cross product
 */
cv::Mat cross_product(const cv::Mat &vec1, const cv::Mat &vec2) {
  cv::Mat cross;
  // Vals in vec 1 
  float vec1_0 = vec1.at<float>(0, 0); 
  float vec1_1 = vec1.at<float>(0, 1); 
  float vec1_2 = vec1.at<float>(0, 2); 
  
  // Vals in vec2 
  float vec2_0 = vec2.at<float>(0, 0); 
  float vec2_1 = vec2.at<float>(0, 1); 
  float vec2_2 = vec2.at<float>(0, 2); 

  // Vals in cross
  float cross_0 = (vec1_1 * vec2_2) - (vec1_2 * vec2_1); 
  float cross_1 = (vec1_2 * vec2_0) - (vec1_0 * vec2_2); 
  float cross_2 = (vec1_0 * vec2_1) - (vec1_1 * vec2_0); 

  cross.at<float>(0, 0) = cross_0; 
  cross.at<float>(0, 1) = cross_1; 
  cross.at<float>(0, 2) = cross_2; 
  
  return cross; 
}

/**
 * @brief Function do an element-wise subtraction of two vectors
 * 
 * @param vec1 one vector to be subtracted
 * @param vec2 other vector to be subtracted
 * @return cv::Mat result of element-wise subtraction
 */
cv::Mat subtract_vectors(const cv::Mat &vec1, const cv::Mat &vec2) {
  cv::Mat out; 
  for(int i = 0; i < vec1.cols; i++) {
    float val = vec1.at<float>(0, i) - vec2.at<float>(0, i); 
    out.at<float>(0, i) = val; 
  }

  return out; 
}

/**
 * @brief Function to combine a vector of matrices into one unified matrix
 * 
 * @param mats matrices to "stack" 
 * @return cv::Mat "stacked" matrices
 */
cv::Mat stack(std::vector<cv::Mat> &mats) {
  cv::Mat out; 

  for(int i = 0; i < mats.size(); i++) {
    cv::Mat curMat = mats[i]; 
    for(int j = 0; j < curMat.cols; j++) {
      out.at<float>(i, j) = curMat.at<float>(0, j); 
    }
  }

  return out; 
}