CXX      = nvcc
CXXFLAGS = -ccbin=g++-4.8 -arch=sm_35 -O3 -std=c++11 -Xcompiler -Wall,-Wextra

TARGET = main
BINS   = $(TARGET)
SRCS   = $(wildcard *.cu)
OBJS   = $(SRCS:%.cu=%.o)
DEPS   = $(OBJS:%.o=%.d)
INCS   = -isystem ../utils
LIBS   =
LNKS   =

RM      = rm -rf

.PHONY: all dep clean

all: $(BINS)
	@ echo > /dev/null

dep: $(DEPS)
	@ echo > /dev/null

run: $(TARGET) test.txt
	./$(TARGET) test.txt

$(BINS): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LIBS) $(LNKS)

%.o: %.cu
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCS)

%.d: %.cu
	@ $(CXX) $(CXXFLAGS) -E --compiler-options -M $< $(INCS) > $@

clean:
	$(RM) $(BINS) $(OBJS) $(DEPS)

-include $(DEPS)
