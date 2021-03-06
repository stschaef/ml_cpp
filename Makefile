CUR_ROOT=/home/stschaef/ml_cpp/
SRC_DIR=$(CUR_ROOT)/src
BIN_DIR=$(CUR_ROOT)/bin
LIB_DIR=$(CUR_ROOT)/lib
TEST_DIR=$(CUR_ROOT)/test
JS_DIR=$(CUR_ROOT)/js

CC=g++
INC=-I/home/stschaef/ml_cpp/include  
PYTHONINCLUDE=-I/usr/include/python3.8/
OPEN_CV=`pkg-config --cflags --libs opencv`
PYTHONLINKS=-L/usr/lib/python3.8/config-3.8-x86_64-linux-gnu/ -lpython3.8
FLAGS=-Wall -Werror -Wextra -g 
LIB=$(wildcard $(LIB_DIR)/*.cpp)
TEST=$(wildcard $(TEST_DIR)/*.cpp)

xor: $(SRC_DIR)/xor.cpp 
	$(CC) $(FLAGS) $(INC) $(PYTHONINCLUDE) $(LIB) $(SRC_DIR)/xor.cpp -o $(BIN_DIR)/xor.o $(PYTHONLINKS) $(OPEN_CV)

iris: $(SRC_DIR)/iris.cpp 
	$(CC) $(FLAGS) $(INC) $(PYTHONINCLUDE) $(LIB) $(SRC_DIR)/iris.cpp -o $(BIN_DIR)/iris.o $(PYTHONLINKS) $(OPEN_CV)

animals: $(SRC_DIR)/animals.cpp 
	$(CC) $(FLAGS) $(INC) $(PYTHONINCLUDE) $(LIB) $(SRC_DIR)/animals.cpp -o $(BIN_DIR)/animals.o $(PYTHONLINKS) $(OPEN_CV)

mnist: $(SRC_DIR)/mnist.cpp 
	$(CC) $(FLAGS) $(INC) $(LIB) $(PYTHONINCLUDE) $(SRC_DIR)/mnist.cpp -o $(BIN_DIR)/mnist.o $(PYTHONLINKS) $(OPEN_CV)

predictor: $(SRC_DIR)/predictor.cpp 
	emcc $(INC) $(filter-out $(LIB_DIR)/utils.cpp, $(LIB)) \
	$(SRC_DIR)/predictor.cpp \
	-Os -g1 \
	-s WASM=1 \
	-s MALLOC=emmalloc \
	-s ALLOW_MEMORY_GROWTH=1 \
	-s EXPORT_ES6=1 \
	-s MODULARIZE=1 \
	-s EXPORT_NAME='createModule' \
	-s 'ENVIRONMENT="web"' \
	--bind \
	-o predictor.js \
	--preload-file "/home/stschaef/ml_cpp/mnist_weights_97.txt" \
	-s ASSERTIONS=1 \
	-s EXPORTED_RUNTIME_METHODS='["ccall","cwrap", "getValue", "setValue"]' \
	-std=c++11 

test: $(TEST)
	$(CC) $(FLAGS) $(INC) -I/usr/include/cppunit $(TEST) -o $(TEST_DIR)/$(NAME).o

clean:
	rm -f $(BIN_DIR)/*.o 
