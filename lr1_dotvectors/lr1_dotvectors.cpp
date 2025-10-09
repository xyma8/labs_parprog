#include <iostream>
#include <Windows.h>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
using namespace std;
using clk = std::chrono::steady_clock; // устойчив к смене системного времени

// Функция генерации вектора n размерности
vector<double> randomVector(size_t n, double minVal = -1000.0, double maxVal = 1000.0) {
    random_device rd;
    mt19937 gen(rd()); // генератор Marsenne Twister
    uniform_real_distribution<> dist(minVal, maxVal); // равномерное распределение

    vector<double> vec(n); // пустой вектор с указанием размерности
    for (size_t i = 0; i < n; ++i) {
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

    for (size_t i = 0; i < vec1.size(); ++i) {
        dot += vec1[i] * vec2[i];
    }

    return dot;
}

double measureExecutionTime(int size) {
    vector<double> a = randomVector(size);
    vector<double> b = randomVector(size);

    auto t0 = clk::now(); // старт измерения времени выполнения

    double result = dotVectors(a, b);

    auto t1 = clk::now(); // окончание измерения времени

    // возвращаем время в микросекундах (мкс)
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

void measureTimeForSizes(int size, int numMeasurements) {
    double seqTotalTime = 0;
    double parTotalTime = 0;

    // Многократные замеры для последовательного алгоритма
    for (int i = 0; i < numMeasurements; ++i) {
        seqTotalTime += measureExecutionTime(size);
    }

    #pragma omp parallel for
    // Многократные замеры для параллельного алгоритма
    for (int i = 0; i < numMeasurements; ++i) {
        parTotalTime += measureExecutionTime(size);
    }

    // Среднее время для каждого из вариантов
    double seqAvgTime = seqTotalTime / numMeasurements;
    double parAvgTime = parTotalTime / numMeasurements;

    // Вывод результатов
    cout << "Размерность: " << size
        << " | Последовательный: " << seqAvgTime << " мкс"
        << " | Параллельный OpenMP: " << parAvgTime << " мкс\n";
}

int main() {
    SetConsoleCP(1251);// установка кодовой страницы win-cp 1251 в поток ввода
    SetConsoleOutputCP(1251); // установка кодовой страницы win-cp 1251 в поток вывода
    //omp_set_num_threads(8);
    int temp;

    cout << "Введите размерность векторов: ";
    if (!(cin >> temp) || temp <= 0) {
        cerr << "Ошибка: размерность должна быть положительным числом" << endl;
        return 1;
    }
    size_t n = static_cast<size_t>(temp);

    cout << "Введите количество прогонов: ";
    if (!(cin >> temp) || temp <= 0) {
        cerr << "Ошибка: кол-во прогонов должно быть положительным числом" << endl;
        return 1;
    }
    size_t numMeasurements = static_cast<size_t>(temp);

    measureTimeForSizes(n, numMeasurements);

    return 0;
}
