# OSX compiler
#CC = clang++

# Dwarf compiler
CC = /Applications/Xcode.app/Contents/Developer/usr/bin/g++

CXX = $(CC)

# OSX include paths (for MacPorts)
#CFLAGS = -I/opt/local/include -I../include

# OSX include paths (for homebrew, probably)
#CFLAGS = -Wc++11-extensions -std=c++11 -I/opt/local/include/opencv4 -I../include -DENABLE_PRECOMPILED_HEADERS=OFF
CFLAGS = -Wc++11-extensions -std=c++11 -I/opt/homebrew/include/opencv4 -I../include -DENABLE_PRECOMPILED_HEADERS=OFF -DWITH_OPENGL=ON

# Dwarf include paths
#CFLAGS = -I../include # opencv includes are in /usr/include
CXXFLAGS = $(CFLAGS)

# OSX Library paths (if you use MacPorts)
#LDFLAGS = -L/opt/local/lib

#OSX Library paths (if you use homebrew, probably)
#LDFLAGS = -L/usr/local/lib
#LDFLAGS = -L/opt/homebrew/lib
LDFLAGS = -L/opt/homebrew/lib -I/opt/homebrew/include

# Dwarf Library paths
#LDFLAGS = -L/opt/local/lib/opencv4/3rdparty -L/opt/local/lib # opencv libraries are here

# opencv libraries
LDLIBS = -ltiff -lpng -ljpeg -llapack -lblas -lz -lwebp -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -framework OpenGL -framework GLUT -lopenimageio


#LDLIBS = -ltiff -lpng -ljpeg -llapack -lblas -lz -ljasper -lwebp -lIlmImf -lgs -framework AVFoundation -framework CoreMedia -framework CoreVideo -framework CoreServices -framework CoreGraphics -framework AppKit -framework OpenCL  -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect

# path to the obj directory
ODIR = obj

# path to the bin directory
BINDIR = ../bin

# preliminary rules to make object files
$(ODIR)/%.o: %.cpp 
	$(CC) $(CFLAGS) -c -o $@ $^ 


# rules to make source files
img: $(ODIR)/imgDisplay.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

#vid: vidDisplay.o filters.o
vid: $(ODIR)/vidDisplay.o $(ODIR)/filters.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

cbir: $(ODIR)/cbir.o $(ODIR)/features.o $(ODIR)/csv_util.o $(ODIR)/filters.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

cbir_db: $(ODIR)/cbir_db.o $(ODIR)/features.o $(ODIR)/csv_util.o $(ODIR)/filters.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

realtime_or: $(ODIR)/realtime_or.o $(ODIR)/filters.o $(ODIR)/recognition.o $(ODIR)/csv_util.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

img_or: $(ODIR)/img_or.o $(ODIR)/filters.o $(ODIR)/recognition.o $(ODIR)/csv_util.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

ar_calibration: $(ODIR)/ar_calibration.o  
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

ar: $(ODIR)/ar.o  
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

ar_extra_features: $(ODIR)/ar_extra_features.o  
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

_06_ar_gl: $(ODIR)/_06_ar_gl.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

_061_template: $(ODIR)/_061_template.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

_03_main: $(ODIR)/_03_main.o $(ODIR)/markerless.o 
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

gather: $(ODIR)/gather_surfaces.o $(ODIR)/ar.o 
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

markerless_main: $(ODIR)/markerless_main.o $(ODIR)/markerless.o $(ODIR)/csv_util.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

checkered: $(ODIR)/checkeredtriangles.o $(ODIR)/imageio.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)


clean:
	rm -f $(ODIR)/*.o *~ 



# face: $(ODIR)/faceDetect.o $(ODIR)/filters.o
# 	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
# 
# imod: $(ODIR)/imgMod.o
# 	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)
# 
# macbeth: $(ODIR)/macbeth.o
# 	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

