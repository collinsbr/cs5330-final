/*
   Brenden Collins and Nate Novak
   CS 5330 Pattern Recognition and Computer Vision - Spring 2022
   Roux Institute - Northeastern University
   Created: 2022/05/02
   Last Modified: 2022/05/02

   This program is the main program for our final project.
   Implements markerless AR to project AR on arbitrary surfaces and
   integrates OpenGL 2 to render much more advanced scenes with graphics
   and lighting.

   Requires:
     - Camera calibration data stored in "calibration.txt"
     - Reference image of desired surface for AR projection

  // TODO:   
   Keypress commands:

   References:
     https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga93efa9b0aa890de240ca32b11253dd4a 
     Skeleton Code for CS290I Homework 1
     2012, Jon Ventura and Chris Sweeney
     https://sites.cs.ucsb.edu/~holl/CS291A/opengl_cv_skeleton.cpp
// TODO: format
// depth testing sources:
//     issue was that I cleared the depth buffer before I loaded in the screen capture from the
//     video feed. Unclear what depth that initialized to, but evidently it was in front of
//     my objects
// https://learnopengl.com/Advanced-OpenGL/Depth-testing
// http://www.songho.ca/opengl/gl_projectionmatrix.html

 */


// GLUT include statements
#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h> // TODO: duplicate?
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include <cstdio>
#include <cstring>
#include <dirent.h>
#include "../include/csv_util.h"
#include "../include/markerless.h"

#define PI 3.1415926

cv::Mat dst;
cv::Mat temp; // cv::Mat objects for frames of video stream
std::ifstream myfile ("calibration.txt"); // object to read files
std::string str_line; // string to read data from file
double d; // temporary variable for reading in data
float f; // temporary variable for reading in data
std::vector<double> data; // vector to hold data read in from file

// source: https://www.cplusplus.com/doc/tutorial/files/

// calibration variables
cv::Mat cam_mat(3, 3, CV_64FC1); 
cv::Mat dist_coef(5, 1, CV_64FC1); 
//cv::Mat cam_mat; // camera matrix
std::vector<float> coeffs = {}; // distortion coefficients
cv::Mat rotations; // rotation vector
cv::Mat translations; // translation vector
cv::Mat out_points; // tranformed image points
cv::Mat rodri; // matrix to hold Rodrigues output
std::vector<cv::Point2f> scene_corners; 

//axis point vector  
std::vector<cv::Vec3f> axis;

// Declare the keypoints and descriptors for the model images
cv::Mat model; 
std::vector<cv::KeyPoint> keypoints_model; 
cv::Mat descriptors_model; 

// create the ORB detector
cv::Ptr<cv::ORB> orb = cv::ORB::create();  


////////////
// original template variables
int w,h;
cv::VideoCapture *cap = NULL;
int width = 640;
int height = 480;
cv::Mat image;
bool drawkps = false; 

/***************************
// texture variables
***************************/
std::vector<std::vector<unsigned char> > pix = {};
static GLuint texName[26];  //texture names
//unsigned char *alphabet[26];

unsigned char tex_a[22500][3];
unsigned char tex_b[22500][3];
unsigned char tex_c[22500][3];
unsigned char tex_d[22500][3];
unsigned char tex_e[22500][3];
unsigned char tex_f[22500][3];
unsigned char tex_g[22500][3];
unsigned char tex_h[22500][3];
unsigned char tex_i[22500][3];
unsigned char tex_j[22500][3];
unsigned char tex_k[22500][3];
unsigned char tex_l[22500][3];
unsigned char tex_m[22500][3];
unsigned char tex_n[22500][3];
unsigned char tex_o[22500][3];
unsigned char tex_p[22500][3];
unsigned char tex_q[22500][3];
unsigned char tex_r[22500][3];
unsigned char tex_s[22500][3];
unsigned char tex_t[22500][3];
unsigned char tex_u[22500][3];
unsigned char tex_v[22500][3];
unsigned char tex_w[22500][3];
unsigned char tex_x[22500][3];
unsigned char tex_y[22500][3];
unsigned char tex_z[22500][3];

// a useful function for displaying your coordinate system
void drawAxes(float length)
{
//  glPushAttrib(GL_POLYGON_BIT | GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT) ;
//
//  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE) ;
//  glDisable(GL_LIGHTING) ;

  glBegin(GL_LINES) ;
  glColor3f(1,0,0) ;
  glVertex3f(0,0,0) ;
  glVertex3f(length,0,0);

  glColor3f(0,1,0) ;
  glVertex3f(0,0,0) ;
  glVertex3f(0,length,0);

  glColor3f(0,0,1) ;
  glVertex3f(0,0,0) ;
  glVertex3f(0,0,length);
  glEnd() ;


//  glPopAttrib() ;
}

// load a texture from the specified file
//int loadTex( unsigned char *loaded_tex, std::string image_path )
//{
//  cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
//  if(img.empty())
//  {
//      std::cout << "Could not read the image: " << image_path << std::endl;
//      return 1;
//  }
//
//  int r = img.rows;
//  int c = img.cols;
//  printf("rows, cols: %d, %d\n", r, c );
//
//  for (int i=0; i<r; i++)
//  {
//    for (int j=0; j<c; j++)
//    {
//      loaded_tex[c*i + j][0] = img.at<cv::Vec3b>(r-i,j)[0];
//      loaded_tex[c*i + j][1] = img.at<cv::Vec3b>(r-i,j)[1];
//      loaded_tex[c*i + j][2] = img.at<cv::Vec3b>(r-i,j)[2];
//    }
//  }
//
//  printf("tex:\n");
//  for (int i=0; i<sizeof(tex)/sizeof(tex[0]); i++)
//  {
//    printf("i: %d, %d %d %d\n", i, tex[i][0], tex[i][1], tex[i][2] );
//  }
//
////  return tex;
//  return 0;
//
//}

void display()
{

    printf("cam mat:\n");
    for (int i=0; i<3; i++)
    {
        for (int j=0; j<3; j++)
        {
            printf("%0.2f ", (float) cam_mat.at<double>(i,j));
        }
        printf("\n");
    }
                    

  int w = dst.size().width;
  int h = dst.size().height;
  int back = 1000;
  int front = 1;
  float k00 = (float) cam_mat.at<double>(0,0);
  float k02 = (float) cam_mat.at<double>(0,2);
  float k11 = (float) cam_mat.at<double>(1,1);
  float k12 = (float) cam_mat.at<double>(1,2);
  float k22 = (float) cam_mat.at<double>(2,2); // should be 1
  int cx0 = 0;
  int cy0 = 0;
  
  float x0 = -1.0, y0 = -1, x1 = 1, y1 = 1, z0 = 1;

  // clear the window
  glClear( GL_COLOR_BUFFER_BIT );

  // show the current camera frame
  //based on the way cv::Mat stores data, you need to flip it before displaying it
  cv::Mat tempimage;
  cv::flip(dst, tempimage, 0);
  glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );
  glClear( GL_DEPTH_BUFFER_BIT );

  //////////////////////////////////////////////////////////////////////////////////
  // Here, set up new parameters to render a scene viewed from the camera.

  //set projection matrix using intrinsic camera params
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
  float intrinsic[16] = 
    { 0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0,
      0, 0, 0, 0 
    };

  // IT WORKS
  // source: http://spottrlabs.blogspot.com/2012/07/opencv-and-opengl-not-always-friends.html

  // hall of failed sources:
  // NOTE: these might actually work, but they either had more complicated steps or slight
  //   underlying differences or implementation hiccups. Regardless, they either were too
  //   complex to try first or they did not work when I tried them (possibly due to errors
  //   of my own making.
  // https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/
  // http://jamesgregson.blogspot.com/2011/11/matching-calibrated-cameras-with-opengl.html
  // http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
  // https://stackoverflow.com/questions/18637494/camera-position-in-world-coordinate-from-cvsolvepnp
  // https://towardsdatascience.com/what-are-intrinsic-and-extrinsic-camera-parameters-in-computer-vision-7071b72fb8ec

  // the stack overflow that led me to all these
  // https://stackoverflow.com/questions/21997021/augmented-reality-openglopencv

  // the template i cribbed from that helped me use OpenGL and OpenCV together
  // https://sites.cs.ucsb.edu/~holl/CS291A/opengl_cv_skeleton.cpp

  intrinsic[0] = 2*k00 / w;
//  intrinsic[2] = (w - 2*k02 * 2*cx0) / w;
  intrinsic[2] = -1 + (2*k02 / w);
  intrinsic[5] = 2*k11 / h;
  intrinsic[6] = -1 + (2*k12 / h);
  intrinsic[10] = (-back - front) / (back - front);
  intrinsic[11] = -2*back*front / (back - front);
  intrinsic[14] = -1;

  glMultTransposeMatrixf( intrinsic );
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

/*
// USE THIS SECTION TO TEST STUFF WITHOUT ATTACHING IT TO THE TARGET
  //set viewport
  glViewport(0, 0, tempimage.size().width, tempimage.size().height);

  //gluPerspective is arbitrarily set, you will have to determine these values based
  //on the intrinsic camera parameters
  gluPerspective(35, tempimage.size().width*1.0/tempimage.size().height, 1, 200); 


  //you will have to set modelview matrix using extrinsic camera params
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(2, 2, -2, 0, 0, 0, 0, 1, 0);  
*/


///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
  // set modelview matrix using camera extrinsic matrix parameters
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  float extrinsic[16] =
    { 1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1 
    };

  // if target area has been project and rotation/translation vectors have been determined
  printf("trans rows: %d, cols: %d, rodri rows: %d, cols: %d\n",
          translations.rows, translations.cols,
          rodri.rows, rodri.cols
        );
  if ( !(translations.rows == 0 && translations.cols == 0) && !(rodri.rows == 0 && rodri.cols == 0) )
  {
      // assign parameters to extrinsic matrix
      extrinsic[0]  = (float)  rodri.at<double>(0,0);
      extrinsic[1]  = (float)  rodri.at<double>(0,1);
      extrinsic[2]  = (float)  rodri.at<double>(0,2);
      extrinsic[3]  = (float)  translations.at<double>(0,0);
      extrinsic[4]  = (float) -rodri.at<double>(1,0);
      extrinsic[5]  = (float) -rodri.at<double>(1,1);
      extrinsic[6]  = (float) -rodri.at<double>(1,2);
      extrinsic[7]  = (float) -translations.at<double>(1,0);
      extrinsic[8]  = (float) -rodri.at<double>(2,0);
      extrinsic[9]  = (float) -rodri.at<double>(2,1);
      extrinsic[10] = (float) -rodri.at<double>(2,2);
      extrinsic[11] = (float) -translations.at<double>(2,0);

  }

  glPushMatrix(); // TODO: do we need this
  // push extrinsic matrix onto transformation stack
  glMultTransposeMatrixf( extrinsic ); 
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

  // create graphical model and perform model transformations 
  glPushMatrix();
//  glRotatef( 20, 0, 1, 0 );
  drawAxes(1.0);

  glPushMatrix();
  glRotatef( 90, 1, 0, 0 );
  glutSolidTeapot(0.1);
  glPopMatrix();

  glPushMatrix();
  glTranslatef( -0.5, -0.25, -0.25 );
  glRotatef( -15, 0, 1, 0 );

  glPushMatrix();
  
  //move to the position where you want the 3D object to go
//  glTranslatef( -0.5, -0.5, 0 );
//  glRotatef( 90, 1, 0, 0 );

  // different GLUT default shapes for testing
//  glutSolidTeapot(0.25);
//    glutWireTeapot(1.5);
//    glutSolidCube(0.1);

  // create 16 cubes in a checkerboard
//  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  float face[6][4][3] =  { 
        {{x0, y0, z0}, {x1, y0, z0}, {x1, y1, z0}, {x0, y1, z0}},	        //front
   	{{x0, y1, -z0}, {x1, y1, -z0}, {x1, y0, -z0}, {x0, y0, -z0}},		//back
       	{{x1, y0, z0}, {x1, y0, -z0}, {x1, y1, -z0}, {x1, y1, z0}},		//right
       	{{x0, y0, z0}, {x0, y1, z0}, {x0, y1, -z0}, {x0, y0, -z0}},		//left
       	{{x0, y1, z0}, {x1, y1, z0}, {x1, y1, -z0}, {x0, y1, -z0}},		//top
       	{{x0, y0, z0}, {x0, y0, -z0}, {x1, y0, -z0}, {x1, y0, z0}}		//bottom
       	};

  for ( int i=0; i<4; i++ )
  {
    for ( int j=0; j<4; j++ )
    {
      glPushMatrix();
      glTranslatef( 0.2*i, -0.2*j, 0.2 );
      glScalef( 0.08, 0.08, 0.08);

      glEnable(GL_TEXTURE_2D);
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
      glEnable( GL_CULL_FACE );
      glCullFace ( GL_BACK );
      for ( int k = 0; k < 6; ++k ) {			//draw cube with texture images
//        glBindTexture(GL_TEXTURE_2D, texName[i]);
//        glBindTexture( GL_TEXTURE_2D, &tex_a );

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D( GL_TEXTURE_2D, 0, 2, 150, 150, 0, GL_BGR, GL_UNSIGNED_BYTE, 
                      tex_a);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        
        glBegin(GL_QUADS);
          glTexCoord2f(0.0, 0.0); glVertex3fv ( face[k][0] );
          glTexCoord2f(1.0, 0.0); glVertex3fv ( face[k][1] );
          glTexCoord2f(1.0, 1.0); glVertex3fv ( face[k][2] );
          glTexCoord2f(0.0, 1.0); glVertex3fv ( face[k][3] );
        glEnd();
      }

      glPopMatrix();
      glFlush();
      glDisable(GL_TEXTURE_2D);
  //  glutSolidCube(0.16);
//      glPopMatrix();

    }
  }
  glPopMatrix();
  glPopMatrix();
  glPopMatrix();
  glPopMatrix();

  // show the rendering on the screen
  glutSwapBuffers();

  // post the next redisplay
  glutPostRedisplay();
}

void reshape( int w, int h )
{
  // set OpenGL viewport (drawable area)
  glViewport( 0, 0, w, h );

//  glMatrixMode(GL_PROJECTION);
//  glLoadIdentity();
//  gluPerspective(80, GLfloat(width)/height, 1, 40);
//  glMatrixMode(GL_MODELVIEW);
//  glLoadIdentity();
//  gluLookAt(2, -1, 5, 0, 0, 0, 0, 1, 0);
////  gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0);
//  glEnable(GL_TEXTURE_2D);


  //glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  // The first element corresponds to the lower left corner of the texture image. 
  // Subsequent elements progress left-to-right through the remaining texels in the lowest row 
  //   of the texture image, and then in successively higher rows of the texture image. The final 
  //   element corresponds to the upper right corner of the texture image.

  //glTexImage2D(GL_TEXTURE_2D,
  //             0,                    // level 0
  //             3,                    // use only R, G, and B components
  //             150, 150,                 // texture has 2x2 texels
  //             0,                    // no border
  //             GL_BGR,               // texels are in BRG format
  //             GL_UNSIGNED_BYTE,     // color components are unsigned bytes
  //             &tex);
  //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

void mouse( int button, int state, int x, int y )
{
  if ( button == GLUT_LEFT_BUTTON && state == GLUT_UP )
    {

    }
}

void keyboard( unsigned char key, int x, int y )
{
  switch ( key )
    {
    case 'q':
      // quit when q is pressed
      exit(0);
      break;

    default:
      break;
    }
  /*
    if(keyEx == 'q')
    {
      break; 
    } 
    else if (keyEx == 's')
    {
      int id = std::rand() % 100; 
      std::string imgname; 
      printf("Please enter the name of the images.\n"); 
      std::cin >> imgname; 
      std::string path = "./out_imgs/" + imgname + ".png"; 
      cv::imwrite(path, dst); 
    }
  */
}

void init()
{
  // assign values to color and location vectors for lights
  GLfloat black[] = { 0.0, 0.0, 0.0, 1.0 };
  GLfloat yellow[] = { 1.0, 1.0, 0.0, 1.0 };
  GLfloat cyan[] = { 0.0, 1.0, 1.0, 1.0 };
  GLfloat white[] = { 1.0, 1.0, 1.0, 1.0 };
  GLfloat position[] = { 2.0, 3.0, -2.0, 0.0 };
  GLfloat dir2[] = { 1.0, -1.0, -1.0, 0.0 };

  // set material parameters
  glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, cyan);
  glMaterialfv(GL_FRONT, GL_SPECULAR, white);
  glMaterialf(GL_FRONT, GL_SHININESS, 30);

  // set light colors for light0
  glLightfv(GL_LIGHT0, GL_AMBIENT, black );
  glLightfv(GL_LIGHT0, GL_DIFFUSE, white ); // yellow
  glLightfv(GL_LIGHT0, GL_SPECULAR, white ); // white
  glLightfv(GL_LIGHT0, GL_POSITION, position );

//  glLightfv(GL_LIGHT1, GL_POSITION, dir2);
//  glLightfv(GL_LIGHT1, GL_DIFFUSE, 
//
//  glLightModeli( GL_LIGHT_MODEL_TWO_SIDE, 0 );

  glEnable(GL_LIGHTING); // so the renderer considers light
  glEnable(GL_LIGHT0); // turn LIGHT0 on
  glEnable(GL_DEPTH_TEST); // so the renderer consider depth
  glDepthFunc( GL_LEQUAL );
//    glDepthRange( 0, 1); // this is the default value

//   glClearColor (0.2, 0.2, 0.8, 0.0);
//   glShadeModel(GL_FLAT);
//
//   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//   //texName is global
//   glGenTextures(6, texName);
//  for ( int i = 0; i < 6; ++i ) {	
//    GLubyte *texImage = makeTexImage( maps[i] );
//    if ( !texImage ) {
//      printf("\nError reading %s \n", maps[i] );
//      continue;
//    }
//    glBindTexture(GL_TEXTURE_2D, texName[i]);		//now we work on texName
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texImageWidth, 
//                texImageHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, texImage);
//
//    delete texImage;					//free memory holding texture image
//  }


}

void idle()
{

  cv::Mat gray;
  *cap >> image; // get a new frame from the camera, treat as a stream
  if( image.empty() )
  {
    printf("frame is empty\n");
    exit(-1);
  }  

  // Convert to grayscale
  cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY); 
  std::vector<cv::KeyPoint> keypoints_scene; 
  cv::Mat descriptors_scene; 
  orb->detectAndCompute( gray, cv::noArray(), keypoints_scene, descriptors_scene );

  // Match the keypoints
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);  
  std::vector<cv::DMatch> acceptable_matches; 
  bool sufficient_matches = false; 
  match_kps(matcher, descriptors_scene, descriptors_model, acceptable_matches, sufficient_matches); 
  image.copyTo(dst); 
  
  if(drawkps)
  {
    std::string winName= "Markerless AR";
    cv::namedWindow(winName, 1);
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
      std::vector<cv::Point2f> scene_corners;

      image.copyTo(dst); 

      get_rots_and_trans(acceptable_matches, keypoints_model, keypoints_scene, model, rotations,
                         translations, cam_mat, coeffs, scene_corners); 

      // Draw the lines betwen the corners (mapped object in the scene)
      cv::line( dst, scene_corners[0],
        scene_corners[1], cv::Scalar(0, 255, 0), 4 );
      cv::line( dst, scene_corners[1],
        scene_corners[2], cv::Scalar( 0, 255, 0), 4 );
      cv::line( dst, scene_corners[2],
        scene_corners[3], cv::Scalar( 0, 255, 0), 4 );
      cv::line( dst, scene_corners[3],
        scene_corners[0], cv::Scalar( 0, 255, 0), 4 );


      // extract rotation matrix from rotation vector using Rodrigues formula 
      cv::Rodrigues( rotations, rodri );

      //Draw the lines betwen the corners (mapped object in the scene)
      //cv::line( dst, scene_corners[0],
      //  scene_corners[1], cv::Scalar(0, 255, 0), 4 );
      //cv::line( dst, scene_corners[1],
      //  scene_corners[2], cv::Scalar( 0, 255, 0), 4 );
      //cv::line( dst, scene_corners[2],
      //  scene_corners[3], cv::Scalar( 0, 255, 0), 4 );
      //cv::line( dst, scene_corners[3],
      //  scene_corners[0], cv::Scalar( 0, 255, 0), 4 );
     
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

    } 
    else
    {
      image.copyTo(dst); 
    }
  }
}

int main( int argc, char **argv )
{
  
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
    
  get_model_kp_desc(orb, model, keypoints_model, descriptors_model); 
  
  // create axis points
  axis.push_back( cv::Vec3f(0,0,0) );
  axis.push_back( cv::Vec3f(1,0,0) );
  axis.push_back( cv::Vec3f(0,1,0) );
  axis.push_back( cv::Vec3f(0,0,1) );

  // get data from file
  cam_mat = cv::Mat::zeros( 3, 3, CV_64FC1 );
  if ( myfile.is_open() )
  {
      // read in camera matrix
      if ( std::getline( myfile, str_line ) )
      {
          data = {}; // clear data vector
          std::istringstream iss(str_line);
          while ( iss >> d ) 
          {
              data.push_back(d);
          }
          for (int i=0; i<3; i++)
          {
              cam_mat.at<double>(i,0) = data[3*i];
              cam_mat.at<double>(i,1) = data[3*i + 1];
              cam_mat.at<double>(i,2) = data[3*i + 2];
          }
      }

      // read in distortion coefficients
      if ( std::getline( myfile, str_line ) )
      {
          coeffs = {}; // coeffs data vector
          std::istringstream iss(str_line);
          while ( iss >> f ) 
          {
              coeffs.push_back(f);
          }
      }
      myfile.close();
  }

  else 
  {
      std::cout << "Unable to open file." << std::endl ;
  }

  // print camera matrix
  printf("camera matrix: \n");
  for (int i=0; i<3; i++)
  {
      for (int j=0; j<3; j++)
      {
          printf("%.2f ", cam_mat.at<double>(i,j) );
      }
      printf("\n");
  }
  printf("end camera matrix\n");

  // print coeffs vector
  printf("coeffs: ");
  for (int i=0; i<coeffs.size(); i++)
  {
      printf("%.4f ", coeffs[i]);
  }
  printf("end coeffs\n");


// BEGIN VIDEO CAPTURE
  if ( argc == 1 ) {
    // start video capture from camera
    cap = new cv::VideoCapture(0);
  } else if ( argc == 2 ) {
    // start video capture from file
    cap = new cv::VideoCapture(argv[1]);
  } else {
    fprintf( stderr, "usage: %s [<filename>]\n", argv[0] );
    return 1;
  }

  // check that video is opened
  if ( cap == NULL || !cap->isOpened() ) {
    fprintf( stderr, "could not start video capture\n" );
    return 1;
  }

  // get width and height
  w = (int) cap->get( cv::CAP_PROP_FRAME_WIDTH );
  h = (int) cap->get( cv::CAP_PROP_FRAME_HEIGHT );
  // On Linux, there is currently a bug in OpenCV that returns 
  // zero for both width and height here (at least for video from file)
  // hence the following override to global variable defaults: 
  width = w ? w : width;
  height = h ? h : height;

  // load texture images
  std::string image_path; 
  int r = 150; // rows in input image
  int c = 150; // cols in input image
  image_path = cv::samples::findFile("./boggle/a.png");
  cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
  for (int i=0; i<r; i++)
  {
    for (int j=0; j<c; j++)
    {
      tex_a[c*i + j][0] = img.at<cv::Vec3b>(r-i,j)[0];
      tex_a[c*i + j][1] = img.at<cv::Vec3b>(r-i,j)[1];
      tex_a[c*i + j][2] = img.at<cv::Vec3b>(r-i,j)[2];
    }
  }

//  loadTex( &tex_a, image_path );


  //alphabet[0]  = &tex_a;
  //alphabet[1]  = &tex_b;
  //alphabet[2]  = &tex_c;
  //alphabet[3]  = &tex_d;
  //alphabet[4]  = &tex_e;
  //alphabet[5]  = &tex_f;
  //alphabet[6]  = &tex_g;
  //alphabet[7]  = &tex_h;
  //alphabet[8]  = &tex_i;
  //alphabet[9]  = &tex_j;
  //alphabet[10] = &tex_k;
  //alphabet[11] = &tex_l;
  //alphabet[12] = &tex_m;
  //alphabet[13] = &tex_n;
  //alphabet[14] = &tex_o;
  //alphabet[15] = &tex_p;
  //alphabet[16] = &tex_q;
  //alphabet[17] = &tex_r;
  //alphabet[18] = &tex_s;
  //alphabet[19] = &tex_t;
  //alphabet[20] = &tex_u;
  //alphabet[21] = &tex_v;
  //alphabet[22] = &tex_w;
  //alphabet[23] = &tex_x;
  //alphabet[24] = &tex_y;
  //alphabet[25] = &tex_z;

    


  // initialize GLUT
  glutInit( &argc, argv );
  //glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE );
  glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH );
  glutInitWindowPosition( 20, 20 );
  glutInitWindowSize( width, height );
  glutCreateWindow( "OpenGL / OpenCV Example" );

  // set up GUI callback functions
  glutDisplayFunc( display );
  glutReshapeFunc( reshape );
  glutMouseFunc( mouse );
  glutKeyboardFunc( keyboard );
  glutIdleFunc( idle );

  // call init()
  init();

  // start GUI loop
  glutMainLoop();

  return 0;
}

//      // print current rotation coefficients 
//      printf("rotation vector: \n");
//      for (int j=0; j < rots.rows; j++ )
//      {
//          for (int k=0; k < rots.cols; k++ )
//          {
//              printf("%lf ", rots.at<double>(j,k) );
//          }
//          printf("\n");
//      }
//      // print transformed Rodrigues coeffs
//      printf("Rodrigues-transformed rotation matrix: \n");
//      for (int j=0; j < rodri.rows; j++ )
//      {
//          for (int k=0; k < rodri.cols; k++ )
//          {
//              printf("%lf ", rodri.at<double>(j,k) );
//          }
//          printf("\n");
//      }
//
//      // print current translation coefficients 
//      printf("translation: \n");
//      for (int j=0; j < trans.rows; j++ )
//      {
//          for (int k=0; k < trans.cols; k++ )
//          {
//              printf("%d, %d: %lf ", j,k, trans.at<double>(j,k) );
//          }
//          printf("\n");
//      }
