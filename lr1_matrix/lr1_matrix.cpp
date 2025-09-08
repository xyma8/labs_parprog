#include <iostream>
#include <Windows.h>
#include <random>
#include <chrono>
using namespace std;
using clk = std::chrono::steady_clock;

int main()
{
    SetConsoleCP(1251);// установка кодовой страницы win-cp 1251 в поток ввода
    SetConsoleOutputCP(1251); // установка кодовой страницы win-cp 1251 в поток вывода
    std::cout << "Hello World!\n";
}
