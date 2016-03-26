CXX      = nvcc
CXXFLAGS = -ccbin=g++-4.8 -arch=sm_35 -O3 -std=c++11 -Xcompiler -Wall,-Wextra

TARGET = main
BINS   = $(TARGET)
SRCS   = $(wildcard *.cu)
HDRS   = $(wildcard *.h)
OBJS   = $(SRCS:%.cu=%.o)
INCS   = -isystem ../utils -isystem $(CUDAROOT)/include
LIBS   =
LNKS   =

RM      = rm -rf

.PHONY: all run clean

all: $(BINS)

run: $(TARGET)
	./$(TARGET)

$(BINS): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LIBS) $(LNKS)

%.o: %.cu $(HDRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCS)

clean:
	$(RM) $(BINS) $(OBJS)
