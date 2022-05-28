all: pgm.o	hough

hough:	houghBase.cu pgm.o
	nvcc houghBase.cu pgm.o -o hough

pgm.o:	pgm.cpp
	g++ -c pgm.cpp -o ./pgm.o
