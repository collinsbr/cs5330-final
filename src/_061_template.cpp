// Skeleton Code for CS290I Homework 1
// 2012, Jon Ventura and Chris Sweeney
// https://sites.cs.ucsb.edu/~holl/CS291A/opengl_cv_skeleton.cpp

// depth testing sources:
//     issue was that I cleared the depth buffer before I loaded in the screen capture from the
//     video feed. Unclear what depth that initialized to, but evidently it was in front of
//     my objects
// https://learnopengl.com/Advanced-OpenGL/Depth-testing
// http://www.songho.ca/opengl/gl_projectionmatrix.html

// adapt the include statements for your system:

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui_c.h>

#ifdef __APPLE__
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#else
#include <GL/gl.h>
#include <GL/glut.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#define PI 3.1415926

using namespace std;
using namespace cv;

#include <cstdio>

cv::Mat frame, temp; // cv::Mat objects for frames of video stream
std::ifstream myfile ("calibration.txt"); // object to read files
std::string str_line; // string to read data from file
double d; // temporary variable for reading in data
float f; // temporary variable for reading in data
std::vector<double> data; // vector to hold data read in from file

// source: https://www.cplusplus.com/doc/tutorial/files/

// chessboard variables
bool foundboard; // bool to hold return value of cv::findChessboardCorners
cv::Size boardsize( 9, 6 ); // what size chessboard to use
std::vector<cv::Point2f> corners, lastcorners; // image points found by findChessboardCorners
std::vector<cv::Vec3f> points; // 3D world points, constructed

// calibration variables
cv::Mat cam_mat; // camera matrix
std::vector<float> coeffs = {}; // distortion coefficients
cv::Mat rots; // rotation vector
cv::Mat trans; // translation vector
cv::Mat out_points; // tranformed image points
cv::Mat rodri; // matrix to hold Rodrigues output

// iterator variables 
int i,j,k;
int c;
int tick;
int r_flag; // rotation flag

//axis point vector  
std::vector<cv::Vec3f> axis;


////////////
// original template variables
int w,h;
cv::VideoCapture *cap = NULL;
int width = 640;
int height = 480;
cv::Mat image;

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

void display()
{

  int w = image.size().width;
  int h = image.size().height;
  int back = 1000;
  int front = 1;
  float k00 = (float) cam_mat.at<double>(0,0);
  float k02 = (float) cam_mat.at<double>(0,2);
  float k11 = (float) cam_mat.at<double>(1,1);
  float k12 = (float) cam_mat.at<double>(1,2);
  float k22 = (float) cam_mat.at<double>(2,2); // should be 1
  int x0 = 0;
  int y0 = 0;

//  printf("w: %d, h: %d\n", w, h);
//  printf("cam mat values: %.1f %.1f %.1f %.1f %.1f\n", k00, k02, k11, k12, k22);

  // clear the window
  glClear( GL_COLOR_BUFFER_BIT );
  //glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  // show the current camera frame
  //based on the way cv::Mat stores data, you need to flip it before displaying it
  cv::Mat tempimage;
  cv::flip(image, tempimage, 0);
  glDrawPixels( tempimage.size().width, tempimage.size().height, GL_BGR, GL_UNSIGNED_BYTE, tempimage.ptr() );
  glClear( GL_DEPTH_BUFFER_BIT );

  //////////////////////////////////////////////////////////////////////////////////
  // Here, set up new parameters to render a scene viewed from the camera.

  //set projection matrix using intrinsic camera params
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();


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
//  intrinsic[2] = (w - 2*k02 * 2*x0) / w;
  intrinsic[2] = -1 + (2*k02 / w);
  intrinsic[5] = 2*k11 / h;
  intrinsic[6] = -1 + (2*k12 / h);
  intrinsic[10] = (-back - front) / (back - front);
  intrinsic[11] = -2*back*front / (back - front);
  intrinsic[14] = -1;

//  for ( i=0; i<16; i++ )
//  {
//      printf("%.2f ", intrinsic[i] );
//  }
//  printf("\n");
        
  glMultTransposeMatrixf( intrinsic );

/*
USE THIS SECTION TO TEST STUFF WITHOUT ATTACHING IT TO THE TARGET
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


  //you will have to set modelview matrix using extrinsic camera params
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  /////////////////////////////////////////////////////////////////////////////////
  // Drawing routine

  //now that the camera params have been set, draw your 3D shapes
  //first, save the current matrix

  float extrinsic[16] =
    { 1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1 
    };

  if ( !(trans.rows == 0 && trans.cols == 0) && !(rodri.rows == 0 && rodri.cols == 0) )
  {
      extrinsic[0]  = (float)  rodri.at<double>(0,0);
      extrinsic[1]  = (float)  rodri.at<double>(0,1);
      extrinsic[2]  = (float)  rodri.at<double>(0,2);
      extrinsic[3]  = (float)  trans.at<double>(0,0);
      extrinsic[4]  = (float) -rodri.at<double>(1,0);
      extrinsic[5]  = (float) -rodri.at<double>(1,1);
      extrinsic[6]  = (float) -rodri.at<double>(1,2);
      extrinsic[7]  = (float) -trans.at<double>(1,0);
      extrinsic[8]  = (float) -rodri.at<double>(2,0);
      extrinsic[9]  = (float) -rodri.at<double>(2,1);
      extrinsic[10] = (float) -rodri.at<double>(2,2);
      extrinsic[11] = (float) -trans.at<double>(2,0);

  }

  glPushMatrix();
  glMultTransposeMatrixf( extrinsic );

  //glRotatef( 30, 1, 1, 0 );
  glPushMatrix();
  drawAxes(1.0);
  glPushMatrix();
  //glRotatef( -90, 1, 0, 0 );
  
  //move to the position where you want the 3D object to go
  //glTranslatef(0, 1, 2); //this is an arbitrary position for demonstration
  glTranslatef( 0, 0, 0 );

  glutSolidTeapot(1.5);
//    glutWireTeapot(1.5);

//  for ( i=0; i<4; i++ )
//  {
//      for ( j=0; j<4; j++ )
//      {
//          glPushMatrix();
//          glTranslatef( i, -j, 2 );
////          glutSolidCube(0.8);
//          glutWireCube(0.8);
//          glPopMatrix();
//      }
//  }


//    glutSolidCube(0.5);
//    glutWireCube(1.5);

//  glutSolidSphere(1.5, 100, 100);
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
}

void init()
{

    GLfloat black[] = { 0.0, 0.0, 0.0, 1.0 };
    GLfloat yellow[] = { 1.0, 1.0, 0.0, 1.0 };
    GLfloat cyan[] = { 0.0, 1.0, 1.0, 1.0 };
    GLfloat white[] = { 1.0, 1.0, 1.0, 1.0 };
//    GLfloat direction[] = { -1.0, -1.0, -1.0, 0.0 };
    GLfloat position[] = { 0.0, 0.0, 2.0, 0.0 };
    GLfloat dir2[] = { 1.0, -1.0, -1.0, 0.0 };

    glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, cyan);
    glMaterialfv(GL_FRONT, GL_SPECULAR, white);
    glMaterialf(GL_FRONT, GL_SHININESS, 30);

    glLightfv(GL_LIGHT0, GL_AMBIENT, black );
    glLightfv(GL_LIGHT0, GL_DIFFUSE, white ); // yellow
    glLightfv(GL_LIGHT0, GL_SPECULAR, white ); // white
    glLightfv(GL_LIGHT0, GL_POSITION, position );

//    glLightfv(GL_LIGHT1, GL_POSITION, dir2);
//    glLightfv(GL_LIGHT1, GL_DIFFUSE, 

//    glLightModeli( GL_LIGHT_MODEL_TWO_SIDE, 0 );

    glEnable(GL_LIGHTING); // so the renderer considers light
    glEnable(GL_LIGHT0); // turn LIGHT0 on
    glEnable(GL_DEPTH_TEST); // so the renderer consider depth
//    glDepthFunc( GL_ALWAYS );
    glDepthFunc( GL_LEQUAL );
//    glDepthRange( 0, 1); // this is the default value
}

void idle()
{
  // grab a frame from the camera
  (*cap) >> image;

  // re-initialize corners and points vectors to empty
  corners = {};
  points = {};

  // check for chessboard
  foundboard = cv::findChessboardCorners( image, boardsize, corners, 
                                          CALIB_CB_ADAPTIVE_THRESH + 
                                          CALIB_CB_FAST_CHECK );
  // if chessboard is found, display axis and AR object based on user input
  if ( foundboard ) 
  {
      cv::cvtColor( image, temp, cv::COLOR_RGB2GRAY, 0 );
      cv::cornerSubPix( temp, corners, cv::Size(11, 11), cv::Size(-1, -1),
                        cv::TermCriteria(cv::TermCriteria::EPS + 
                                         cv::TermCriteria::MAX_ITER, 30, 0.1) );
      lastcorners = corners; // copy corners to be used later

      // assign 3D world points
      for (i=0; i < boardsize.height; i++)
      {
          for (j=0; j < boardsize.width; j++)
          {
              points.push_back( cv::Vec3f( j, -i, 0 ) );
          }
      }
      // solve image coordinates of world points given camera pose
      solvePnP( points, corners, cam_mat, coeffs, rots, trans );

      cv::Rodrigues( rots, rodri );

      // print current rotation coefficients 
      printf("rotation vector: \n");
      for ( j=0; j < rots.rows; j++ )
      {
          for ( k=0; k < rots.cols; k++ )
          {
              printf("%lf ", rots.at<double>(j,k) );
          }
          printf("\n");
      }
      // print transformed Rodrigues coeffs
      printf("Rodrigues-transformed rotation matrix: \n");
      for ( j=0; j < rodri.rows; j++ )
      {
          for ( k=0; k < rodri.cols; k++ )
          {
              printf("%lf ", rodri.at<double>(j,k) );
          }
          printf("\n");
      }

      // print current translation coefficients 
      printf("translation: \n");
      for ( j=0; j < trans.rows; j++ )
      {
          for ( k=0; k < trans.cols; k++ )
          {
              printf("%d, %d: %lf ", j,k, trans.at<double>(j,k) );
          }
          printf("\n");
      }

      // draw x-y-z axis at origin
      projectPoints( axis, rots, trans, cam_mat, coeffs, out_points ); // project axis pts
      cv::Point oo = Point( out_points.at<cv::Vec2f>(0,0) );
      cv::circle( image, oo, 6, {255, 0, 255} );
      cv::circle( image, oo, 8, {255, 0, 255} );
      cv::Point ox = Point( out_points.at<cv::Vec2f>(1,0) );
      cv::Point oy = Point( out_points.at<cv::Vec2f>(2,0) );
      cv::Point oz = Point( out_points.at<cv::Vec2f>(3,0) );
      cv::arrowedLine( image, oo, ox, { 0, 0, 255 }, 2);
      cv::arrowedLine( image, oo, oy, { 0, 255, 0 }, 2 );
      cv::arrowedLine( image, oo, oz, { 255, 0, 0 }, 2 );
  }
}

int main( int argc, char **argv )
{
  
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
          for (i=0; i<3; i++)
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
      cout << "Unable to open file.";
  }

  // print camera matrix
  printf("camera matrix: \n");
  for ( i=0; i<3; i++ )
  {
      for ( j=0; j<3; j++ )
      {
          printf("%.2f ", cam_mat.at<double>(i,j) );
      }
      printf("\n");
  }
  printf("end camera matrix\n");

  // print coeffs vector
  printf("coeffs: ");
  for (i=0; i<coeffs.size(); i++)
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
  // gluLookAt params:
  // camera center x, y, z = Cx Cy Cz
  // target point x, y, z = px, py, pz
  // up direction x, y, z = ux, uy, uz

  // Extrinsic matrix is:
  // s1 s2 s3
  // u1' u2' u3'
  // -L1 -L2 -L3

//  float s1, s2, s3; //
//  float u1, u2, u3;
//  float L1, L2, L3;

