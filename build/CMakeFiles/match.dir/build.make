# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.22

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\CMake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\CMake\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\natej\Documents\GitHub\cs5330-final

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\natej\Documents\GitHub\cs5330-final\build

# Include any dependencies generated for this target.
include CMakeFiles/match.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/match.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/match.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/match.dir/flags.make

CMakeFiles/match.dir/src/match_keypoints.cpp.obj: CMakeFiles/match.dir/flags.make
CMakeFiles/match.dir/src/match_keypoints.cpp.obj: CMakeFiles/match.dir/includes_CXX.rsp
CMakeFiles/match.dir/src/match_keypoints.cpp.obj: ../src/match_keypoints.cpp
CMakeFiles/match.dir/src/match_keypoints.cpp.obj: CMakeFiles/match.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\natej\Documents\GitHub\cs5330-final\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/match.dir/src/match_keypoints.cpp.obj"
	C:\Users\natej\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\x86_64-w64-mingw32-g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/match.dir/src/match_keypoints.cpp.obj -MF CMakeFiles\match.dir\src\match_keypoints.cpp.obj.d -o CMakeFiles\match.dir\src\match_keypoints.cpp.obj -c C:\Users\natej\Documents\GitHub\cs5330-final\src\match_keypoints.cpp

CMakeFiles/match.dir/src/match_keypoints.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/match.dir/src/match_keypoints.cpp.i"
	C:\Users\natej\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\x86_64-w64-mingw32-g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\natej\Documents\GitHub\cs5330-final\src\match_keypoints.cpp > CMakeFiles\match.dir\src\match_keypoints.cpp.i

CMakeFiles/match.dir/src/match_keypoints.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/match.dir/src/match_keypoints.cpp.s"
	C:\Users\natej\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin\x86_64-w64-mingw32-g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\natej\Documents\GitHub\cs5330-final\src\match_keypoints.cpp -o CMakeFiles\match.dir\src\match_keypoints.cpp.s

# Object files for target match
match_OBJECTS = \
"CMakeFiles/match.dir/src/match_keypoints.cpp.obj"

# External object files for target match
match_EXTERNAL_OBJECTS =

../bin/match.exe: CMakeFiles/match.dir/src/match_keypoints.cpp.obj
../bin/match.exe: CMakeFiles/match.dir/build.make
../bin/match.exe: C:/Users/natej/opencv/build/x64/mingw/lib/libopencv_gapi452.dll.a
../bin/match.exe: C:/Users/natej/opencv/build/x64/mingw/lib/libopencv_highgui452.dll.a
../bin/match.exe: C:/Users/natej/opencv/build/x64/mingw/lib/libopencv_ml452.dll.a
../bin/match.exe: C:/Users/natej/opencv/build/x64/mingw/lib/libopencv_objdetect452.dll.a
../bin/match.exe: C:/Users/natej/opencv/build/x64/mingw/lib/libopencv_photo452.dll.a
../bin/match.exe: C:/Users/natej/opencv/build/x64/mingw/lib/libopencv_stitching452.dll.a
../bin/match.exe: C:/Users/natej/opencv/build/x64/mingw/lib/libopencv_video452.dll.a
../bin/match.exe: C:/Users/natej/opencv/build/x64/mingw/lib/libopencv_videoio452.dll.a
../bin/match.exe: C:/Users/natej/opencv/build/x64/mingw/lib/libopencv_dnn452.dll.a
../bin/match.exe: C:/Users/natej/opencv/build/x64/mingw/lib/libopencv_imgcodecs452.dll.a
../bin/match.exe: C:/Users/natej/opencv/build/x64/mingw/lib/libopencv_calib3d452.dll.a
../bin/match.exe: C:/Users/natej/opencv/build/x64/mingw/lib/libopencv_features2d452.dll.a
../bin/match.exe: C:/Users/natej/opencv/build/x64/mingw/lib/libopencv_flann452.dll.a
../bin/match.exe: C:/Users/natej/opencv/build/x64/mingw/lib/libopencv_imgproc452.dll.a
../bin/match.exe: C:/Users/natej/opencv/build/x64/mingw/lib/libopencv_core452.dll.a
../bin/match.exe: CMakeFiles/match.dir/linklibs.rsp
../bin/match.exe: CMakeFiles/match.dir/objects1.rsp
../bin/match.exe: CMakeFiles/match.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\natej\Documents\GitHub\cs5330-final\build\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ..\bin\match.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\match.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/match.dir/build: ../bin/match.exe
.PHONY : CMakeFiles/match.dir/build

CMakeFiles/match.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\match.dir\cmake_clean.cmake
.PHONY : CMakeFiles/match.dir/clean

CMakeFiles/match.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\natej\Documents\GitHub\cs5330-final C:\Users\natej\Documents\GitHub\cs5330-final C:\Users\natej\Documents\GitHub\cs5330-final\build C:\Users\natej\Documents\GitHub\cs5330-final\build C:\Users\natej\Documents\GitHub\cs5330-final\build\CMakeFiles\match.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/match.dir/depend
