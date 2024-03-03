INC_DIR=-I /home/intern/mmy/utils/gcc-11.4.0/include
LIB_DIR=-L /home/intern/mmy/utils/gcc-11.4.0/lib64

.PHONY:all clean

object=main

GCC11_PATH=/home/intern/mmy/utils/gcc-11.4.0

all:
	$(GCC11_PATH)/bin/g++ -o $(object) $(object).cpp -std=c++20

clean:
	rm -rf $(object)