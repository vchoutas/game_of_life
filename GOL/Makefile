TARGET = game_of_life
# compiler =$(shell echo $(CXX))
compiler=nvcc
CXXFLAGS =-Wall -Wextra -O3
LDFLAGS=-lcuda -lcudart
NVCCFlags = -arch=sm_20 -Xcompiler "-Wall -Wextra -O3"
filename = table500x500.bin
N = 500
maxGen = 100

all: main.o serial.o utilities.o simple_cuda.o many_cuda.o
	$(compiler) -o $(TARGET).out report_main.o utilities.o serial.o simple_cuda.o many_cuda.o
test: test.o serial.o
	$(compiler) -o $(TARGET)_test.exe test.o serial.o
main.o: report_main.cpp
	$(compiler) -c report_main.cpp  $(NVCCFlags)
test.o: test.cpp test.h
	$(compiler) -c test.cpp $(NVCCFlags)
serial.o: serial.cpp serial.h
	$(compiler) -c serial.cpp $(NVCCFlags)
simple_cuda.o: simple_cuda.cu simple_cuda.cuh
	$(compiler) -c simple_cuda.cu $(NVCCFlags)
many_cuda.o: many_cuda.cu many_cuda.cuh
	$(compiler) -c many_cuda.cu $(NVCCFlags)
utilities.o: utilities.cu utilities.cuh
	$(compiler) -c utilities.cu $(NVCCFlags)
run: $(TARGET).out
	./$(TARGET).out $(filename) $(N) $(maxGen)
clean:
	rm -f *~ *.o *.out

