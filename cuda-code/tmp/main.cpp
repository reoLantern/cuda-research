#include <iostream>
#include <bit>

int main()
{
    uint32_t value = 0x40490FDB; // 用uint32_t保存的浮点数，值为3.14159

    float *floatPtr = reinterpret_cast<float *>(&value); // 使用reinterpret_cast进行类型转换
    float floatValue = *floatPtr; // 通过解引用操作读取浮点数的值
    std::cout << "Float value: " << floatValue << std::endl;

    float staticCast_value = static_cast<float>(value);
    std::cout << "Static_cast float value: " << staticCast_value << std::endl;

    float bitcast_value = std::bit_cast<float>(value);
    std::cout << "Bit_cast float value: " << bitcast_value << std::endl;

    return 0;
}
