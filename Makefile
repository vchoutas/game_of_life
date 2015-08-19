TARGET = game_of_life
compiler = clang++
N = 500
filename = table500x500.bin


all: main.o game_of_life.o utilities.o
	$(compiler) -o $(TARGET).out main.o game_of_life.o utilities.o -lGLU -lglut -lGL
main.o: main.cpp game_of_life.h
	$(compiler) -c main.cpp -Wall -Wextra
game_of_life.o : game_of_life.cpp utilities.h
	$(compiler) -c game_of_life.cpp -Wall -Wextra
utilities.o : utilities.cpp
	$(compiler) -c utilities.cpp -Wall -Wextra
run: $(TARGET)
	./$(TARGET).out $(N) $(filename)
clean:
	rm -f *~ *.o *.out

