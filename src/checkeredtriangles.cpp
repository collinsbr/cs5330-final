// This application is a trivial illustration of texture mapping.  It draws
// several triangles, each with a texture mapped on to it.  The same texture
// is used for each triangle, but the mappings vary quite a bit so it looks as
// if each triangle has a different texture.

// SOURCE: https://cs.lmu.edu/~ray/notes/openglexamples/

#ifdef __APPLE_CC__
#define GL_SILENCE_DEPRECATION
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <cstdlib>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <png.h>

#include "imageio.hpp"


// Define a 2 x 2 red and yellow checkered pattern using RGB colors.
#define red {0xff, 0x00, 0x00}
#define yellow {0xff, 0xff, 0x00}
#define magenta {0xff, 0, 0xff}

std::vector<std::vector<unsigned char> > pix = {};

unsigned char loaded_tex[22500][3];

GLubyte texture[][3] = {
    red, magenta,
    yellow, red,
};

Glubyte *texImage;

// Fixes up camera and remaps texture when window reshaped.
void reshape(int width, int height) {
  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(80, GLfloat(width)/height, 1, 40);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(2, -1, 5, 0, 0, 0, 0, 1, 0);
//  gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0);
  glEnable(GL_TEXTURE_2D);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  // The first element corresponds to the lower left corner of the texture image. 
  // Subsequent elements progress left-to-right through the remaining texels in the lowest row 
  //   of the texture image, and then in successively higher rows of the texture image. The final 
  //   element corresponds to the upper right corner of the texture image.
  glTexImage2D(GL_TEXTURE_2D,
               0,                    // level 0
               3,                    // use only R, G, and B components
//               2, 2,                 // texture has 2x2 texels
//               128, 128,                 // texture has 2x2 texels
               150, 150,                 // texture has 2x2 texels
               0,                    // no border
//               GL_RGB,               // texels are in RGB format
               GL_BGR,               // texels are in BRG format
               GL_UNSIGNED_BYTE,     // color components are unsigned bytes
               texImage);
//               &loaded_tex);
//               texture);

//  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
//  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
}

// Draws three textured triangles.  Each triangle uses the same texture,
// but the mappings of texture coordinates to vertex coordinates is
// different in each triangle.
void display() {
  glClear(GL_COLOR_BUFFER_BIT);
  glBegin(GL_TRIANGLES);
    glTexCoord2f(0.5, 1.0);    glVertex2f(-3, 3);
    glTexCoord2f(0.0, 0.0);    glVertex2f(-3, 0);
    glTexCoord2f(1.0, 0.0);    glVertex2f(0, 0);

    glTexCoord2f(1.0, 1.0);        glVertex2f(3, 3);
    glTexCoord2f(0.0, 0.0);    glVertex2f(0, 0);
    glTexCoord2f(1.0, 0.0);      glVertex2f(3, 0);
//    glTexCoord2f(4, 8);        glVertex2f(3, 3);
//    glTexCoord2f(0.0, 0.0);    glVertex2f(0, 0);
//    glTexCoord2f(8, 0.0);      glVertex2f(3, 0);

    glTexCoord2f(5, 5);        glVertex2f(0, 0);
    glTexCoord2f(0.0, 0.0);    glVertex2f(-1.5, -3);
    glTexCoord2f(4, 0.0);      glVertex2f(1.5, -3);
  glEnd();
  glFlush();
}

// Initializes GLUT and enters the main loop.
int main(int argc, char** argv)
{


//  std::string image_path = cv::samples::findFile("./boggle/a.png");
//  cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
//  if(img.empty())
//  {
//      std::cout << "Could not read the image: " << image_path << std::endl;
//      return 1;
//  }
////  cv::imshow("Display window", img);
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
//  printf("tex size: %lu\n", pix.size() );
//  for (int i=0; i<sizeof(loaded_tex)/sizeof(loaded_tex[0]); i++)
//  {
//    printf("i: %d, %d %d %d\n", i, loaded_tex[i][0], loaded_tex[i][1], loaded_tex[i][2] );
//  }

  char *loadfile = "./boggle./a.png";

  texImage = loadImageRGBA( (char *) loadfile, &width, &height);



  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
  glutInitWindowSize(520, 390);
  glutCreateWindow("Textured Triangles");
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutMainLoop();
}



//  OIIO::ImageInput *in = OIIO::ImageInput::open ("a.png");
//
//  const OIIO::ImageSpec &spec = in->spec();
//  int xres = spec.width;
//  int yres = spec.height;
//  int channels = spec.nchannels;
//  std::vector<unsigned char> pixels (xres*yres*channels);
//  in->read_image (TypeDesc::UINT8, &pixels[0]);
//  in->close();
//  OIIO::ImageInput::destroy (in);

