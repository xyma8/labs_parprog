#include <mpi.h>
//#include <Windows.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
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

// Метод подсчета времени выполнения последовательного выполнения
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
        double dt = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        seqTotalTime += dt;
    }

    double seqAvgTime = seqTotalTime / numMeasurements;
    cout << left << setw(25) << " | Последовательный:"
        << left << seqAvgTime << " мкс; "
        << "dot=" << dot << endl;
}

// Метод подсчета времени выполнения параллельного выполнения
void measureTimePar(vector<double>& vecA, vector<double>& vecB, int numMeasurements) {

}

// Метод подсчета времени выполнения параллельного с оптимизацией выполнения
void measureTimeParOpt(vector<double>& vecA, vector<double>& vecB, int numMeasurements) {

}

int main() {
    //SetConsoleCP(1251);// установка кодовой страницы win-cp 1251 в поток ввода
    //SetConsoleOutputCP(1251); // установка кодовой страницы win-cp 1251 в поток вывода

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    size_t n = 0, m = 0, numMeasurements = 0;
    vector<double> vectorA;
    vector<double> vectorB;

    if (rank==0) {
        int temp;
        cout << "Введите количество строк: ";
        cin >> temp;
        n = temp;

        cout << "Введите количество столбцов: ";
        cin >> temp;
        m = temp;

        cout << "Введите количество прогонов: ";
        cin >> temp;
        numMeasurements = temp;

        random_device rd;
        vectorA = randomVector(n, 12345);
        vectorB = randomVector(n, 54321);
    }

    // Передаем значения переменных всем процессам
    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numMeasurements, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    cout << endl << "Результаты " << numMeasurements << " прогонов" << " по алгоритмам:" << endl;
    measureTimeSeq(vectorA,vectorB, numMeasurements);
    measureTimePar(vectorA,vectorB, numMeasurements);
    measureTimeParOpt(vectorA, vectorB, numMeasurements);

    return 0;
}
