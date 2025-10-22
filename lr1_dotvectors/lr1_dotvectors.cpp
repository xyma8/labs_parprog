
#include <iostream>
#include <Windows.h>
#include <vector>
#include <random>
#include <chrono>
#include <omp.h>
#include <iomanip>  // для setw, left, right
using namespace std;
using clk = std::chrono::steady_clock; // устойчив к смене системного времени

// Функция генерации вектора n размерности
vector<double> randomVector(size_t n, unsigned int seed = 12345, double minVal = -1000.0, double maxVal = 1000.0) {
    mt19937 gen(seed); // генератор Marsenne Twister
    uniform_real_distribution<> dist(minVal, maxVal); // равномерное распределение

    vector<double> vec(n); // пустой вектор с указанием размерности
    for (size_t i = 0; i < n; ++i) {
        vec[i] = dist(gen);
    }

    return vec;
}

void measureTimeSeq(vector<double>& vecA, vector<double>& vecB, int numMeasurements) {
    double seqTotalTime = 0;

    double dot = 0.0;

    for (int i = 0; i < numMeasurements; ++i) {
        dot = 0;
        auto t0 = clk::now();

        for (size_t i = 0; i < vecA.size(); ++i) {
            dot += vecA[i] * vecB[i];
        }

        auto t1 = clk::now();
        static volatile double sink; // защищаем от выкидывания
        sink = dot;

        double dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        seqTotalTime += dt;
    }

    double seqAvgTime = seqTotalTime / numMeasurements;
    cout << left << setw(25) << " | Последовательный:"
        << left << seqAvgTime << " мкс; "
        << "dot=" << dot << endl;
}

void measureTimePar(vector<double>& vecA, vector<double>& vecB, int numMeasurements) {
    double parTotalTime = 0;
    double t0 = 0.0; // ОБЩЕЕ время для всех потоков
    double dot = 0.0;
    size_t size = vecA.size();

    #pragma omp parallel
    {
        // Многократные замеры для параллельного алгоритма
        for (int i = 0; i < numMeasurements; ++i) {
            #pragma omp barrier

            #pragma omp single
            {
                dot = 0;
                t0 = omp_get_wtime();
            }

            double local_dot = 0;
            #pragma omp for
            for (int i = 0; i < vecA.size(); ++i) {
                local_dot += vecA[i] * vecB[i];
            }

            #pragma omp critical
            dot += local_dot;

            #pragma omp single
            {
                double dt = (omp_get_wtime() - t0) * 1e6; // умножаем для микросекунд
                parTotalTime += dt;
                static volatile double sink; sink = dot; // не даём выкинуть
            }
        }
    }

    double parAvgTime = parTotalTime / numMeasurements;
    cout << left << setw(25) << " | Параллельный:"
        << left << parAvgTime << " мкс; "
        << "dot=" << dot << endl;
}

void measureTimeParOpt(vector<double>& vecA, vector<double>& vecB, int numMeasurements) {
    double parOptTotalTime = 0;
    double t0 = 0.0; // ОБЩЕЕ время для всех потоков
    double dot = 0.0;
    size_t size = vecA.size();

    #pragma omp parallel
    {
        // Многократные замеры для параллельного алгоритма
        for (int i = 0; i < numMeasurements; ++i) {
        #pragma omp barrier

            #pragma omp single
            {
                dot = 0;
                t0 = omp_get_wtime();
            }

            #pragma omp for reduction(+:dot) schedule(static)
            for (int i = 0; i < vecA.size(); ++i) {
                dot += vecA[i] * vecB[i];
            }

            #pragma omp single
            {
                double dt = (omp_get_wtime() - t0) * 1e6; // умножаем для микросекунд
                parOptTotalTime += dt;
                static volatile double sink; sink = dot; // не даём выкинуть
            }
        }
    }

    double parAvgTimeOpt = parOptTotalTime / numMeasurements;
    cout << left << setw(25) << " | Пар.(Оптимизация):"
        << left << parAvgTimeOpt << " мкс; "
        << "dot=" << dot << endl;
}

int main() {
    SetConsoleCP(1251);// установка кодовой страницы win-cp 1251 в поток ввода
    SetConsoleOutputCP(1251); // установка кодовой страницы win-cp 1251 в поток вывода

    // фиксируем число потоков, чтобы измерения были стабильны
    omp_set_dynamic(0);
    omp_set_num_threads(omp_get_max_threads());
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

    random_device rd;

    unsigned int seedA = rd();
    cout << "Генератор вектора A (seed) = " << seedA << endl;
    vector<double> vectorA = randomVector(n, seedA);

    unsigned int seedB = rd();
    cout << "Генератор вектора B (seed) = " << seedB << endl;
    vector<double> vectorB = randomVector(n, seedB);

    cout << endl << "Результаты " << numMeasurements << " прогонов" << " по алгоритмам:" << endl;
    measureTimeSeq(vectorA,vectorB, numMeasurements);
    measureTimePar(vectorA,vectorB, numMeasurements);
    measureTimeParOpt(vectorA, vectorB, numMeasurements);

    return 0;
}
