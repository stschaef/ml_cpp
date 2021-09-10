NAME=ml_cpp

TOP_DIR=~/ml_cpp
SRC_DIR=$(TOP_DIR)/src
BIN_DIR=$(TOP_DIR)/bin
LIB_DIR=$(TOP_DIR)/lib
TEST_DIR=$(TOP_DIR)/test

CC=g++
INC=-Iinclude -I/usr/include/cppunit
FLAGS=-Wall -Werror -Wextra -g 
DEBUG_FLAGS=-g -fsanitize=address
LIB=$(wildcard $(LIB_DIR)/*.cpp)
TEST=$(wildcard $(TEST_DIR)/*.cpp)


all: $(NAME)

$(NAME): $(LIB)
	$(CC) $(FLAGS) $(INC) $(LIB) -o $(BIN_DIR)/$(NAME).o

.PHONY: test

test: $(TEST)
	$(CC) $(FLAGS) $(INC) $(TEST) -o $(TEST_DIR)/$(NAME).o

clean:
	rm -f $(BIN_DIR)/*.o 
