CC = g++
CFLAGS = -c

main: main.cpp
	$(CC) $(CFLAGS) main.cpp -I SFML/include
	$(CC) main.o -o sfml-app -L SFML/lib -lsfml-graphics -lsfml-window -lsfml-system
	export LD_LIBRARY_PATH=SFML/lib && ./sfml-app
