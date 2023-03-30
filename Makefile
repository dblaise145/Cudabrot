CC = g++
CFLAGS = -c

mandelbrot: mandelbrot.cpp
	$(CC) $(CFLAGS) mandelbrot.cpp -I SFML/include
	$(CC) mandelbrot.o -o sfml-app -L SFML/lib -lsfml-graphics -lsfml-window -lsfml-system
	export LD_LIBRARY_PATH=SFML/lib && ./sfml-app
