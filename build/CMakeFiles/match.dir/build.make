# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/nate/Documents/Spring22/cs5330-final

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/nate/Documents/Spring22/cs5330-final/build

# Include any dependencies generated for this target.
include CMakeFiles/match.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/match.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/match.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/match.dir/flags.make

CMakeFiles/match.dir/src/match_keypoints.cpp.o: CMakeFiles/match.dir/flags.make
CMakeFiles/match.dir/src/match_keypoints.cpp.o: ../src/match_keypoints.cpp
CMakeFiles/match.dir/src/match_keypoints.cpp.o: CMakeFiles/match.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nate/Documents/Spring22/cs5330-final/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/match.dir/src/match_keypoints.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/match.dir/src/match_keypoints.cpp.o -MF CMakeFiles/match.dir/src/match_keypoints.cpp.o.d -o CMakeFiles/match.dir/src/match_keypoints.cpp.o -c /home/nate/Documents/Spring22/cs5330-final/src/match_keypoints.cpp

CMakeFiles/match.dir/src/match_keypoints.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/match.dir/src/match_keypoints.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nate/Documents/Spring22/cs5330-final/src/match_keypoints.cpp > CMakeFiles/match.dir/src/match_keypoints.cpp.i

CMakeFiles/match.dir/src/match_keypoints.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/match.dir/src/match_keypoints.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nate/Documents/Spring22/cs5330-final/src/match_keypoints.cpp -o CMakeFiles/match.dir/src/match_keypoints.cpp.s

CMakeFiles/match.dir/src/markerless.cpp.o: CMakeFiles/match.dir/flags.make
CMakeFiles/match.dir/src/markerless.cpp.o: ../src/markerless.cpp
CMakeFiles/match.dir/src/markerless.cpp.o: CMakeFiles/match.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nate/Documents/Spring22/cs5330-final/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/match.dir/src/markerless.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/match.dir/src/markerless.cpp.o -MF CMakeFiles/match.dir/src/markerless.cpp.o.d -o CMakeFiles/match.dir/src/markerless.cpp.o -c /home/nate/Documents/Spring22/cs5330-final/src/markerless.cpp

CMakeFiles/match.dir/src/markerless.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/match.dir/src/markerless.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nate/Documents/Spring22/cs5330-final/src/markerless.cpp > CMakeFiles/match.dir/src/markerless.cpp.i

CMakeFiles/match.dir/src/markerless.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/match.dir/src/markerless.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nate/Documents/Spring22/cs5330-final/src/markerless.cpp -o CMakeFiles/match.dir/src/markerless.cpp.s

CMakeFiles/match.dir/src/csv_util.cpp.o: CMakeFiles/match.dir/flags.make
CMakeFiles/match.dir/src/csv_util.cpp.o: ../src/csv_util.cpp
CMakeFiles/match.dir/src/csv_util.cpp.o: CMakeFiles/match.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/nate/Documents/Spring22/cs5330-final/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/match.dir/src/csv_util.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/match.dir/src/csv_util.cpp.o -MF CMakeFiles/match.dir/src/csv_util.cpp.o.d -o CMakeFiles/match.dir/src/csv_util.cpp.o -c /home/nate/Documents/Spring22/cs5330-final/src/csv_util.cpp

CMakeFiles/match.dir/src/csv_util.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/match.dir/src/csv_util.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/nate/Documents/Spring22/cs5330-final/src/csv_util.cpp > CMakeFiles/match.dir/src/csv_util.cpp.i

CMakeFiles/match.dir/src/csv_util.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/match.dir/src/csv_util.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/nate/Documents/Spring22/cs5330-final/src/csv_util.cpp -o CMakeFiles/match.dir/src/csv_util.cpp.s

# Object files for target match
match_OBJECTS = \
"CMakeFiles/match.dir/src/match_keypoints.cpp.o" \
"CMakeFiles/match.dir/src/markerless.cpp.o" \
"CMakeFiles/match.dir/src/csv_util.cpp.o"

# External object files for target match
match_EXTERNAL_OBJECTS =

../bin/match: CMakeFiles/match.dir/src/match_keypoints.cpp.o
../bin/match: CMakeFiles/match.dir/src/markerless.cpp.o
../bin/match: CMakeFiles/match.dir/src/csv_util.cpp.o
../bin/match: CMakeFiles/match.dir/build.make
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_alphamat.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_barcode.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_intensity_transform.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_mcc.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_rapid.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_wechat_qrcode.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.5.4d
../bin/match: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.5.4d
../bin/match: CMakeFiles/match.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/nate/Documents/Spring22/cs5330-final/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ../bin/match"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/match.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/match.dir/build: ../bin/match
.PHONY : CMakeFiles/match.dir/build

CMakeFiles/match.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/match.dir/cmake_clean.cmake
.PHONY : CMakeFiles/match.dir/clean

CMakeFiles/match.dir/depend:
	cd /home/nate/Documents/Spring22/cs5330-final/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/nate/Documents/Spring22/cs5330-final /home/nate/Documents/Spring22/cs5330-final /home/nate/Documents/Spring22/cs5330-final/build /home/nate/Documents/Spring22/cs5330-final/build /home/nate/Documents/Spring22/cs5330-final/build/CMakeFiles/match.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/match.dir/depend

