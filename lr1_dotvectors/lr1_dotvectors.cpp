
#include <iostream>
#include <Windows.h>
#include <vector>
#include <sstream>
#include <string>
#include <random>
#include <chrono>
using namespace std;
using clk = std::chrono::steady_clock; // устойчив к смене системного времени

// Функция считывания вектора из вводимой строки
vector<double> readVector() {
    string line;
    getline(cin, line); //читаем всю строку
    stringstream ss(line);

    vector<double> vec;
    string token;
    while (ss >> token) {
        try {
            double num = stod(token); // конвертирование строки в double
            vec.push_back(num);
        }
        catch (const invalid_argument&) {
            cerr << "'" << token << "'" << " не число" << endl;
        }
    }

    return vec;
}

// Функция генерации вектора n размерности
vector<double> randomVector(size_t n, double minVal = -10.0, double maxVal = 10.0) {
    random_device rd;
    mt19937 gen(rd()); // генератор Marsenne Twister
    uniform_real_distribution<> dist(minVal, maxVal); // равномерное распределение

    vector<double> vec(n); // пустой вектор с указанием размерности
    for (size_t i = 0; i < n; i++) {
        vec[i] = dist(gen);
    }

    return vec;
}

// Функция вычисления скалярного произведения двух векторов
double dotVectors(vector<double> vec1, vector<double> vec2) {
    if (vec1.size() != vec2.size()) {
        cout << "Ошибка: векторы разной размерности" << endl;
        //throw invalid_argument("Векторы разной размерности");
    }

    double dot = 0;

    for (size_t i = 0; i < vec1.size(); i++) {
        dot += vec1[i] * vec2[i];
    }

    return dot;
}

int main() {
    SetConsoleCP(1251);// установка кодовой страницы win-cp 1251 в поток ввода
    SetConsoleOutputCP(1251); // установка кодовой страницы win-cp 1251 в поток вывода

    /*
    cout << "Введите первый вектор через пробел: ";
    vector<double> a = readVector();

    cout << "Введите второй вектор через пробел: ";
    vector<double> b = readVector();
    */

    int temp;
    cout << "Введите размерность векторов: ";
    if (!(cin >> temp) || temp <= 0) {
        cerr << "Ошибка: размерность должна быть положительным числом" << endl;
        return 1;
    }

    size_t n = static_cast<size_t>(temp);

    vector<double> a = randomVector(n);
    /*
    cout << "Вектор a = ";
    for (double x : a) {
        cout << x << ", ";
    }
    cout << endl;
    */

    vector<double> b = randomVector(n);
    /*
    cout << "Вектор b = ";
    for (double x : a) {
        cout << x << ", ";
    }
    cout << endl;
    */

    auto t0 = clk::now(); // старт измерения времени выполнения

    double result = dotVectors(a, b);
    cout << "Скалярное произведение = " << result << endl;

    auto t1 = clk::now(); // окончание измерения времени выполнения

    cout << "Время выполнения скалярного умножения: " << endl;
    // вычисление времени выполнения (миллисекунды)
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    cout << ms << " мс" << endl;

    // вычисление времени выполнения (микроскунды)
    auto mcs = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    cout << mcs << " мкс" << endl;

    return 0;
}
