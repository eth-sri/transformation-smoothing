#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "domains/interval.h"
#include "utils/hyperbox.h"
#include "utils/image.h"
#include <string>
// #include <vector>
#include <sstream>
#include <iostream>
#include <fstream>
#include <tuple>
#include <stack>
#include <Python.h>
#include <numpy/arrayobject.h> // numpy
#include <queue>
#include <mutex>
#include <algorithm>
#include <thread>
#include <queue>
#include <atomic>
#include <indicators/progress_bar.hpp>

using namespace indicators;

#pragma once

typedef std::chrono::high_resolution_clock clock_;
typedef std::chrono::duration<double, std::ratio<1> > second_;

struct arg_t {
  Image img;
  size_t initialSplits;
  Transform transform;
  float target;
  bool inv;
  HyperBox params;
  vector<vector<float>> betas;
  size_t resize_postTransform; 
  size_t center_crop_postTranform;
  float filter_sigma;
  size_t filter_size;
  float radiusDecrease;
  bool debug;
  size_t batch_size;
  size_t threads;
  bool gpu;
  float stopErr;
  size_t refinements;
  bool doInt;
  size_t timeout; // in seconds
  arg_t(){};
};

typedef tuple<float, vector<float>, HyperBox> split_t; //previous max error, previous errors, parameters


enum Status {
  stop, initialSplits, splits
};

class Statistics {

private:
  const arg_t& args;
  const float areaTodo;
  atomic<float> areaDone;
  atomic<size_t> splits_count;
  atomic<size_t> splits_done;
  atomic<size_t> initial_splits_done;
  atomic<float> targetErr;
  vector<float> err;
  mutex mtx_err;
  vector<float> maxParam;
  mutex mtx_maxParam; 
  atomic<Status> status;
  chrono::time_point<clock_> start;
  ProgressBar bar;
  size_t non_empty_initial_splits;
  vector<function<void()>>  after_initial_callback_fns;
  mutex mtx_update_status;
  
public:
  Statistics(const arg_t& args) : args(args),
                                  areaTodo(args.params.area()),
                                  err(args.betas[0].size(), 0),
                                  maxParam(args.betas.size(), 0),
                                  bar(option::BarWidth{50},
                                      option::Start{" ["},
                                      option::Fill{"█"},
                                      option::Lead{"█"},
                                      option::Remainder{"-"},
                                      option::End{"]"},
                                      option::PrefixText{""},
                                      option::ForegroundColor{Color::green},
                                      option::ShowElapsedTime{true},
                                      option::ShowRemainingTime{true},
                                      option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}){
    areaDone.store(0);
    splits_count.store(0);
    initial_splits_done.store(0);
    splits_done.store(0);
    targetErr.store(args.target);
    status.store(Status::initialSplits);
    start = clock_::now();
    non_empty_initial_splits = 0;
  };


  void register_after_initial_callback(function<void()> fn) {
    after_initial_callback_fns.push_back(fn);
  }
  
  vector<float> get_err() {
    return err;
  };

  vector<float> get_maxParam() {
    return maxParam;
  };
  
  void add_area(const float area) {
    float currentAreaDone = areaDone.load();
    while(!areaDone.compare_exchange_weak(currentAreaDone, currentAreaDone + area)) {}
  };  


  void split_done() {
    splits_done += 1;
  };

  void initial_split_done() {
    initial_splits_done += 1;
  };
  
  void add_split_count(const size_t cnt) {
    splits_count += cnt;
  };
  
  void update_maxParam(const HyperBox& split) {
    mtx_maxParam.lock();
    for(size_t u = 0; u < maxParam.size(); ++u) {
      maxParam[u] = max(maxParam[u], abs(split.it[u]).sup);
    }
    mtx_maxParam.unlock();
  };

  void update_err(const vector<float>& epsilons) {
    mtx_err.lock();
    for(size_t u = 0; u < err.size(); ++u) {
      err[u] = max(err[u], epsilons[u]);
    }
    mtx_err.unlock();
  };  

  void update_targetErr(const float maxErr) {
    float currentTarget = targetErr.load();
    while(!targetErr.compare_exchange_weak(currentTarget, maxErr)) {}
  };  

  float get_targetErr() {
    return targetErr.load();
  };

  Status get_status() {
    return status.load();
  };
  
  void update_status() {
    mtx_update_status.lock();
    if (get_status() == Status::initialSplits && initial_splits_done.load() >= args.initialSplits) {
      non_empty_initial_splits = splits_count.load();
      for (auto fn : after_initial_callback_fns) fn();
      status.store(Status::splits);
    }
    else if (get_status() == Status::splits && splits_done.load() >= splits_count.load()) {
      status.store(Status::stop);
    }   
    
    double elapsedSeconds = std::chrono::duration_cast<second_> (clock_::now() - start).count();
    if (get_targetErr() > args.stopErr || elapsedSeconds >= args.timeout) {
      status.store(Status::stop);
    }
    mtx_update_status.unlock();    
  };

  bool running() {
    return get_status() != Status::stop;
  };
  
  bool is_initial() {
    return get_status() == Status::initialSplits;
  };

  void update_progressbar(const float maxSplitErr) {
    float percentage = 100.0f * areaDone.load() / areaTodo;
    bar.set_progress(percentage);
    ostringstream oss;
    oss << "[" << min(areaDone.load(), areaTodo) << "/" << areaTodo << "] ";
    string msg;
    if (is_initial()) {
      oss << "[" << initial_splits_done << "/" << args.initialSplits << "] ";
      msg = "processing initial splits";
    } else {
      oss << "[" << splits_done << "/" << splits_count << "] ";
      msg = "processing further splits";
    }

    if(get_targetErr() >= args.stopErr) {
      msg = "refinement limit hit";
    };
    
    oss << "[" << targetErr << "] ";
    mtx_err.lock();
    oss << "[" << max(maxSplitErr, *max_element(err.begin(), err.end())) << "]";
    mtx_err.unlock();

    oss << " " << non_empty_initial_splits << "/" << args.initialSplits << " non-empty initial";     
    oss << " | " << msg;

    bar.set_option(option::PostfixText{oss.str()});
    if (!running()) bar.mark_as_completed();  
  };
};


void initGeometricTools();

tuple<vector<float>,
      vector<float>> getE(const arg_t& args);

void process_initial_split(const arg_t& args,
                           split_t& split_,
                           Image& image,
                           Statistics& stats,
                           vector<split_t>& out_subsplits);

void process_split(const arg_t& args,
                   split_t& split_,
                   Image& image,
                   float* filter,
                   Statistics& stats,
                   vector<split_t>& out_subsplits);
  
bool calc(const arg_t& args,
          const Image& img,
          const HyperBox& gamma,
          const float *filter,
          vector<float>& out_epsilons);
