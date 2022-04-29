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
 * @brief Function to multiply each value in a matrix by 01
 * 
 * @param matrix is the matrix to be multiplied by -1
 */
void invert_values(cv::Mat &matrix); 

/**
 * @brief Function to multiply two matrices
 * 
 * @param mat1 One of the matrices to multiply
 * @param mat2 The other matrix to multiply 
 * @return cv::Mat result of the mulitplication 
 */
cv::Mat matrix_multiplication(cv::Mat &mat1, cv::Mat &mat2); 

/**
 * @brief Function to calculate the multiplicative inverse of a matrix
 * 
 * @param matrix matrix to be inverted
 * @return cv::Mat result of inversion
 */
cv::Mat multiplicative_inverse(cv::Mat &matrix);

/**
 * @brief Get the column of a specific matrix as a 1D Mat
 * 
 * @param matrix matrix from which to retrieve the column
 * @param colnum the column number
 * @return cv::Mat the output
 */
cv::Mat get_column(const cv::Mat &matrix, const int colnum); 

/**
 * @brief Function to calculate the L2 norm of the vectors 
 * 
 * @param vector vector to determine the L2 norm of 
 * @return float result of the L2 norm calculation
 */
float l2_norm(const cv::Mat vector); 

/**
 * @brief Function to normalize a vector based on a normalization factor
 * 
 * @param vector vector to be normalized
 * @param norm_factor factor by which to normalize a vector
 * @return cv::Mat vector that holds normalized values
 */
cv::Mat normalize_vector(const cv::Mat &vector, const float norm_factor); 

/**
 * @brief Function do an element-wise addition of two vectors
 * 
 * @param vec1 one vector to be added
 * @param vec2 other vector to be added
 * @return cv::Mat result of element-wise addition
 */
cv::Mat add_vectors(const cv::Mat &vec1, const cv::Mat &vec2); 

/**
 * @brief Function to calculate the cross product of 2 3X1 vectors
 * 
 * @param vec1 one vectors of the cross product
 * @param vec2 the other vector of the cross product
 * @return cv::Mat the output vector of the cross product
 */
cv::Mat cross_product(const cv::Mat &vec1, const cv::Mat &vec2); 

/**
 * @brief Function do an element-wise subtraction of two vectors
 * 
 * @param vec1 one vector to be subtracted
 * @param vec2 other vector to be subtracted
 * @return cv::Mat result of element-wise subtraction
 */
cv::Mat subtract_vectors(const cv::Mat &vec1, const cv::Mat &vec2); 

/**
 * @brief Function to combine a vector of matrices into one unified matrix
 * 
 * @param mats matrices to "stack" 
 * @return cv::Mat "stacked" matrices
 */
cv::Mat stack(std::vector<cv::Mat> &mats); 