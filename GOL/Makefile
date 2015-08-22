TARGET = game_of_life
compiler = g++
N = 500
filename = table500x500.bin


all: main.o game_of_life.o serial.o utilities.o 
	$(compiler) -o  $(TARGET) main.o game_of_life.o  utilities.o serial.o

test:test.o serial.o 
	$(compiler) -o test test.o serial.o

run: $(TARGET)
	./$(TARGET) $(N) $(filename)

clean:
	rm -f *~ *.o

