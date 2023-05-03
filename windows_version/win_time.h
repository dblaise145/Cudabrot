#ifndef CUDA_LAB_TIMER_H
#define CUDA_LAB_TIMER_H

#include <chrono>

std::chrono::time_point<std::chrono::system_clock> start, end;
#define START_TIMER(X) start = std::chrono::system_clock::now();
#define STOP_TIMER(X) end = std::chrono::system_clock::now(); \
    std::chrono::duration<double> _timer_ ## X = end - start;  
#define GET_TIMER(X) (_timer_ ## X).count()

#endif //CUDA_LAB_TIMER_H