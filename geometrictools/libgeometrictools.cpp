#include "libgeometrictools.h"
#include "gpu/gpu.h"
 
void initGeometricTools() {
  initGPU();
}

bool calc(const arg_t& args,
          const Image& img,
          const HyperBox& gamma,
          const float *filter,
          vector<float>& out_epsilons) {
  assert(gamma.it[0].sup - gamma.it[0].inf > Constants::EPS);
  const size_t batch_size = (img.isGPU()) ? args.batch_size : 1;
  size_t dims = 0;
  if (args.transform == Transform::rotation) dims = 1;
  else if (args.transform == Transform::translation) dims = 2;
  else assert(false);
  
  Image imgGamma;
  Image imgOrig;
  bool isEmpty = false;
  const size_t M = args.betas[0].size();


  //compute inverse
  if (args.inv) {
    if (args.debug) cout << "start inverse" << endl << flush;    
    if (args.transform == Transform::rotation) {
      imgOrig = img.inverseRotation(gamma.it[0], isEmpty, args.refinements, args.doInt, args.doInt);
    }
    else if (args.transform == Transform::translation) {
      imgOrig = img.inverseTranslation(gamma.it[0], gamma.it[1], isEmpty, args.refinements, args.doInt, args.doInt);
    }
    
    imgGamma = img;    
    if (args.debug) cout << "done inverse; empty: " << isEmpty << endl << flush;
    if (isEmpty) {
      for(size_t k = 0; k < M; ++k) {
        out_epsilons.push_back(0);
      }
      return true;
    }
    
  } else {    
    imgOrig = img;
    if (args.transform == Transform::rotation) {
      assert(img.img);
      assert(imgOrig.img);
      imgGamma = imgOrig.rotate(gamma.it[0], args.doInt);
    }
    else if (args.transform == Transform::translation)
      imgGamma = imgOrig.translate(gamma.it[0], gamma.it[1], args.doInt);       
  }
  
  const size_t nChannels = img.nChannels();
  const size_t N = ceil(((double)M) / batch_size); // number of loop passes    
  for (size_t k = 0; k < N; ++k) {
    vector<vector<float>> beta_batch(dims);
    vector<vector<Interval>> beta_gamma_batch(dims);    
    vector<float> epsilon_batch;
    for(size_t j = 0; j < batch_size; ++j) {
      const size_t m = k + j * N;
      if (m < M) {
        for(size_t dim = 0;  dim < dims; ++dim) {
          beta_batch[dim].push_back(args.betas[dim][m]);
          beta_gamma_batch[dim].push_back(gamma.it[dim] + args.betas[dim][m]);
        }        
      }
    }
    const size_t B = beta_batch[0].size();
    
    if (args.debug) cout << "start transform" << endl << flush;

    Image transformed_gamma_transformed_beta_processed, transformed_gamma_beta_processed;

    if (args.transform == Transform::rotation) {
      transformed_gamma_transformed_beta_processed = imgGamma.rotate(beta_batch[0], args.doInt);
      transformed_gamma_beta_processed = imgOrig.rotate(beta_gamma_batch[0], args.doInt);
    }
    else if (args.transform == Transform::translation) {
      transformed_gamma_transformed_beta_processed = imgGamma.translate(beta_batch[0], beta_batch[1], args.doInt);
      transformed_gamma_transformed_beta_processed = transformed_gamma_transformed_beta_processed.rect_vignetting((int)args.radiusDecrease);
      transformed_gamma_beta_processed = imgOrig.translate(beta_gamma_batch[0], beta_gamma_batch[1], args.doInt);
      transformed_gamma_beta_processed = transformed_gamma_beta_processed.rect_vignetting((int)args.radiusDecrease);
    }       
    if (args.debug) cout << "done transform" << endl << flush;
    
    assert(transformed_gamma_transformed_beta_processed.nChannels() == nChannels * B);
    assert(transformed_gamma_beta_processed.nChannels() == nChannels * B);
      
    if (args.resize_postTransform > 0 || args.center_crop_postTranform > 0) {
      if (args.debug) cout << "start post" << endl << flush;
      transformed_gamma_transformed_beta_processed = transformed_gamma_transformed_beta_processed.resize(args.resize_postTransform, args.doInt, args.center_crop_postTranform);
      transformed_gamma_beta_processed = transformed_gamma_beta_processed.resize(args.resize_postTransform, args.doInt, args.center_crop_postTranform);
      if (args.debug) cout << "done post" << endl << flush;
    }

    assert(transformed_gamma_transformed_beta_processed.nChannels() == nChannels * B);
    assert(transformed_gamma_beta_processed.nChannels() == nChannels * B);
      
    if (args.debug) {
      transformed_gamma_transformed_beta_processed.to_cpu().channels(0, nChannels).saveBMP("transformed_gamma_transformed_beta_post_resize_inf.bmp", true);
      transformed_gamma_transformed_beta_processed.to_cpu().channels(0, nChannels).saveBMP("transformed_gamma_transformed_beta_post_resize_sup.bmp", false);
    }

    if (args.debug) cout << "start norm 123" << endl << flush;
    epsilon_batch = transformed_gamma_transformed_beta_processed.filterVignettingL2diffC(transformed_gamma_beta_processed,
                                                                                         filter,
                                                                                         args.filter_size,
                                                                                         (args.transform == Transform::rotation) ? args.radiusDecrease : -1,
                                                                                         nChannels);
    out_epsilons.insert(out_epsilons.end(),epsilon_batch.begin(),epsilon_batch.end());
    if (args.debug) cout << "done norm" << endl << flush;
  }

  assert(out_epsilons.size() == M);

  return false;
}


void process_initial_split(const arg_t& args,
                           split_t& split_,
                           Image& image,
                           Statistics& stats,
                           vector<split_t>& out_subsplits) {
  out_subsplits.push_back(split_);
}

void process_split(const arg_t& args,
                   split_t& split_,
                   Image& image,
                   float* filter,
                   Statistics& stats,
                   vector<split_t>& out_subsplits) {
  const float prevMaxErr = get<0>(split_);
  vector<float> prevErr = get<1>(split_);
  HyperBox split = get<2>(split_);
  if (args.debug) cout << "Split " << split << " came from interval with error " << prevMaxErr;
  
  bool atRefinementLimit = false;
  // if the split is smaller than the minimum size (this can happen after refining at the borders)
  // make it large enough (but only once -- in order to avoid infinite refinements)
  if (split.anyDimBelow(Constants::EPS)) {
    if (args.debug) cout << " as it is currently smaller than the minimum it is reshaped into ";
    for(size_t d = 0; d < split.dim; ++d) {
      Interval it = split.it[d];
      float upper = it.sup;
      float k = 0.25f;
      while(upper - it.inf <= Constants::EPS) {upper = it.sup + k * Constants::EPS; k*=2;}
      split.it[d] = Interval(it.inf, upper);
      atRefinementLimit = true;
    }
    if (args.debug) cout << split << " ";
  }
  // now all dimensions of the split are eat least the minimum size  
  assert(!split.anyDimBelow(Constants::EPS));

  // compute error
  vector<float> epsilons;
  bool isEmpty = calc(args, image, split, filter, epsilons);
  
  // update statistics and add new splits
  if (isEmpty) {
    stats.add_area(split.area());
  } else {          
    float maxErr = *max_element(epsilons.begin(), epsilons.end());
    if (args.debug) cout << " has err " << maxErr; 
    if (maxErr <= stats.get_targetErr() || atRefinementLimit ) { // no further refinement
      if (args.debug) cout << "-> done" << endl << flush;
      stats.add_area(split.area());
      stats.update_maxParam(split);
      stats.update_err(epsilons);        
      if (atRefinementLimit) stats.update_targetErr(maxErr);
    } else { // refinement
      if (args.debug) cout << "and will be refined into: ";
      float current = split.maxDim();
      float next = (current + Constants::EPS) / 4.0f;
      size_t s = ceilf(current/next);
      vector<HyperBox>subsplits = split.split(s);
      for(HyperBox b : subsplits) out_subsplits.push_back({maxErr, epsilons, b});
    }
  }
}


tuple<vector<float>,
      vector<float>> getE(const arg_t& args) {
  // check input
  size_t dims = 0;
  switch(args.transform) {
  case Transform::rotation: dims = 1; break;
  case Transform::translation: dims = 2; break;
  default: assert(false);
  }      
  assert(args.betas.size() == dims); // we have dims x n args.betas
  assert(args.betas.size() > 0);
  size_t n = args.betas[0].size();
  for(auto b : args.betas) assert(b.size() == n);
  
  // create images and filters for each thread
  vector<Image> images(args.threads);
  vector<float*> filters(args.threads);

  int numGPUs = 0;
  if (args.gpu) {
    cudaGetDeviceCount(&numGPUs);    
  }
  
  for(size_t k = 0; k < args.threads; ++k) {
    if (args.gpu) {
      images[k] = args.img.to_gpu(k % numGPUs);
    } else {
      images[k] = args.img;
    }

    if (args.filter_sigma > 0) {
      filters[k] = images[k].getFilter(args.filter_sigma, args.filter_size);
    } else {
      filters[k] = 0;
    }
  }

  if(args.debug) cout.precision(22);

  Statistics stats(args); 
  auto comp = [] (split_t &a, split_t &b) -> bool {
    const float errA = get<0>(a);
    const float errB = get<0>(b);      
    return errA < errB; //return True if a < b
  };

  priority_queue<split_t,
                 vector<split_t>,
                 decltype(comp)> splits(comp);

  vector<HyperBox> initialSplits_ = args.params.split(args.initialSplits);
  //assert(initialSplits_.size() == args.initialSplits); args.initialSplits^nDims
  vector<split_t> initialSplits;
  for(HyperBox s : initialSplits_) initialSplits.push_back({numeric_limits<float>::infinity(), {}, s});
  mutex mtx_splits;  

  // define worker threads
  auto worker = [&](const size_t tid) {
    split_t split;
    while(stats.running()){
      bool hasWork = false;
      mtx_splits.lock();
      if (stats.is_initial()) {
        hasWork = !initialSplits.empty();
        if (hasWork) {
          split = initialSplits.back();
          initialSplits.pop_back();
        }
      }
      else {
        hasWork = !splits.empty();
        if (hasWork) {
          split = splits.top();
          splits.pop();
        }
      }
      mtx_splits.unlock();

      if (hasWork) {
        vector<split_t> subsplits;        
        if (stats.is_initial()){
          process_initial_split(args, split, images[tid], stats, subsplits);
          stats.initial_split_done();
        } else {
          process_split(args, split, images[tid], filters[tid], stats, subsplits);
          stats.split_done();
        }
        
        // add subsplits
        mtx_splits.lock();
        for (auto b : subsplits) splits.push(b);
        mtx_splits.unlock();
        stats.add_split_count(subsplits.size());
        stats.update_status();        
      } else {
        this_thread::sleep_for(chrono::milliseconds(200));
      }
    }
  };

  // start worker thread
  vector<thread> workers(args.threads);
  for (size_t k = 0; k < args.threads; ++k){
    workers[k] = thread(worker, k);
  }

  // monitor threads   
  while(stats.running()) {
    this_thread::sleep_for(chrono::milliseconds(200));
    const float maxSplitErr = get<0>(splits.top());
    stats.update_status(); // by also calling this here we make sure that the timeout is enforced
    if (!args.debug) stats.update_progressbar(maxSplitErr);
  }
  stats.update_progressbar(0);
  
  for (size_t k = 0; k < args.threads; ++k){
    workers[k].join();
  }

  vector<float> err = stats.get_err();
  vector<float> maxParam = stats.get_maxParam();
  while(!splits.empty())  {
    vector<float> splitErr = get<1>(splits.top());
    splits.pop();
    for(size_t u = 0; u < err.size(); ++u) {
      err[u] = max(err[u], splitErr[u]);
    }
  }
  
  return {err, maxParam};
}
