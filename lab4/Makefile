CXX      = nvcc
CXXFLAGS = -ccbin=g++-4.8 -O3 -std=c++11 -Xcompiler -Wall,-Wextra

SRCS = $(sort $(wildcard *.cu))
TGTS = $(SRCS:%.cu=%.ptx)

RM   = rm -rf

.PHONY: all run clean

all: $(TGTS)

lab4a.ptx: lab4a.cu
	$(CXX) -arch=sm_20 $(CXXFLAGS) -ptx $< -o $@
	ptxas $@ -v -arch=sm_20
	@ echo "================================================================"

lab4b.ptx: lab4b.cu
	$(CXX) -arch=sm_20 $(CXXFLAGS) -ptx $< -o $@
	ptxas $@ -v -arch=sm_20
	@ echo "================================================================"

lab4c.ptx: lab4c.cu
	$(CXX) -arch=sm_50 $(CXXFLAGS) -ptx $< -o $@
	ptxas $@ -v -arch=sm_50
	@ echo "================================================================"

lab4d.ptx: lab4d.cu
	$(CXX) -arch=sm_50 $(CXXFLAGS) -ptx $< -o $@
	ptxas $@ -v -arch=sm_50
	@ echo "================================================================"

lab4e.ptx: lab4e.cu
	$(CXX) -arch=sm_50 $(CXXFLAGS) -ptx $< -o $@
	ptxas $@ -v -arch=sm_50
	@ echo "================================================================"

clean:
	$(RM) $(TGTS)
