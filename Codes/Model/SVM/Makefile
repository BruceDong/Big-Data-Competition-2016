# C++ compiler and flags
# for both Linux and MAC
CXX 		= g++
CXXFLAGS 	= -Wall -ansi -g

# C compiler and flags

# For Linux
#CC         = cc 

# For MAC
CC			= gcc
CCFLAGS		= -g

# Final output name
PROGRAM		= SVR

# All files to be compiled and linked
SRCDIR		= './'
BUILDDIR	= 

# Get C files 
C_SRCS	:= $(shell find $(SRCDIR) -name '*.c')
C_OBJS := $(C_SRCS:.c=.o)

# Get C++ files
CPP_SRCS := $(shell find $(SRCDIR) -name '*.cpp')
CPP_OBJS := ${CPP_SRCS:.cpp=.o}

# This is simply a list of all the ".o" files, both from C and C++ source files.
OBJS := $(C_OBJS) $(CPP_OBJS)

# This is a place holder. If you used program_INCLUDE_DIRS := ./include, then headers in "./include" would be found with #include <>
program_INCLUDE_DIRS :=


# This is a place holder. If you used program_LIBRARY_DIRS := ./lib, then libraries in "./lib" would be found by the linker.
program_LIBRARY_DIRS :=

# This is a place holder. If you used program_LIBRARIES := boost_signals, then libboost_signals would be linked in.
program_LIBRARIES :=


# Link all obj files 
$(PROGRAM): $(OBJS) 
	$(CXX) -o $@  $(OBJS)

# C++ files compile
%.o: %.cpp 
	$(CXX) -o $@ -c $(CXXFLAGS) $<

# C files compile
%.o: %.c 
	$(CC) -o $@ -c $(CCFLAGS) $<

# clean ups
clean:
	rm -rf $(PROGRAM) $(OBJS)


