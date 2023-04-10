CC = g++
CFLAGS = -c

all: mandelbrot_serial mandelbrot_cuda

mandelbrot_serial: mandelbrot_serial.cpp
	$(CC) $(CFLAGS) mandelbrot_serial.cpp -I SFML/include
	$(CC) mandelbrot_serial.o -o mandel_serial -L SFML/lib -lsfml-graphics -lsfml-window -lsfml-system

mandelbrot_cuda: mandelbrot_cuda.cu
	nvcc mandelbrot_cuda.cu -c -I SFML/include
	nvcc mandelbrot_cuda.o -o mandel_cuda -L SFML/lib -lsfml-graphics -lsfml-window -lsfml-system