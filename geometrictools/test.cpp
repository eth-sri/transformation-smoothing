#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "utils/image.h"
#include "gpu/gpu.h"
#include <cuda_runtime_api.h>
#include <random>
#include "utils/hyperbox.h"
#include "libgeometrictools.h"

std::default_random_engine random_generator;
size_t random_repeats = 5;
int seed = 0;

int main( int argc, char* argv[] ) {
  // global setup...
  initGPU();

  int result = Catch::Session().run( argc, argv );

  // global clean-up...
  return result;
}

TEST_CASE( "rotation is inside inverse rotation, no integer errs", "[inv rot]" ) {
  random_generator.seed(seed);
  std::uniform_real_distribution<float> distribution(-1,1);
  Image img(2, 2, 1, "x, 0, 0, 1, 0");

  float gamma = 0.1;
  float delta = 0.01;

  for(size_t k = 0; k < random_repeats; ++k) {  
    for (bool toInt : { false, true })
      for (bool refineToInt : { false, true })
        for (int refinements : { 0, 1, 5, 10, 50 }) {
          if (toInt && k > 0) continue; //toInt test only works if the input image has inf==sup, which is not the case for the random images
          Image imgR = img.rotate(gamma, toInt);
          bool isEmpty;
          Image imgRinv = imgR.inverseRotation(Interval(gamma - delta, gamma + delta), isEmpty, refinements, toInt, refineToInt);
          REQUIRE( !isEmpty );
          REQUIRE( imgRinv.contains(img) );
        }
    img.randomize(random_generator);
    gamma = distribution(random_generator);
  }  
}

TEST_CASE( "inv rotation behaves the same on CPU and GPU", "[inv rot]" ) {
  random_generator.seed(seed);
  std::uniform_real_distribution<float> distribution(-1,1);

  int numGPUs = 0;
  cudaGetDeviceCount(&numGPUs);
  
  Image img(2, 2, 1, "x, 0, 0, 1, 0");
  float gamma = 0.1;
  float delta = 0.01;

  for(size_t k = 0; k < random_repeats; ++k) {  
    for(int gpu = 0; gpu < numGPUs; ++gpu) {
      for (bool toInt : { false, true })
        for (bool refineToInt : { false, true })
          for (int refinements : { 0, 1, 5, 10, 50 }) { 
            if (toInt && k > 0) continue; //toInt test only works if the input image has inf==sup, which is not the case for the random images
            Image imgR = img.rotate(gamma, toInt);
            bool isEmpty;
            Image imgRinv = imgR.inverseRotation(Interval(gamma - delta, gamma + delta), isEmpty, refinements, toInt, refineToInt);
            bool isEmptyG;
            Image imgRinvG = imgR.to_gpu(gpu).inverseRotation(Interval(gamma - delta, gamma + delta), isEmptyG, refinements, toInt, refineToInt).to_cpu();
            REQUIRE( !isEmpty );
            REQUIRE( !isEmptyG );
            if(!imgRinv.equals(imgRinvG, 1e-5)) {
              cout << img << endl;
              cout << gamma << endl;
              cout << imgR << endl;
              cout << imgRinv << endl;
              cout << imgRinvG << endl;
            }
            REQUIRE( imgRinv.equals(imgRinvG, 1e-5) );
          }
    }
    img.randomize(random_generator);
    gamma = distribution(random_generator);
  }  
}


TEST_CASE( "inv rotation with vingette behaves the same on CPU and GPU", "[inv rot]" ) {
  random_generator.seed(seed);
  std::uniform_real_distribution<float> distribution(-1,1);

  int numGPUs = 0;
  cudaGetDeviceCount(&numGPUs);
  
  Image img(2, 2, 1, "x, 0, 0, 1, 0");
  float gamma = 0.1;
  float delta = 0.01;

  for(size_t k = 0; k < random_repeats; ++k) {  
    for(int gpu = 0; gpu < numGPUs; ++gpu) {
      for (bool toInt : { false, true })
        for (bool refineToInt : { false, true })
          for (int refinements : { 0, 1, 5, 10, 50 }) { 
            if (toInt && k > 0) continue; //toInt test only works if the input image has inf==sup, which is not the case for the random images
            Image imgR = img.rotate(gamma, toInt);
            bool isEmpty;
            tuple<Image, Image> out = imgR.inverseRotationWithVingette(Interval(gamma - delta, gamma + delta), isEmpty, refinements, toInt, refineToInt);
            Image imgRinv = get<0>(out);
            Image vingette = get<1>(out);
            bool isEmptyG;
            tuple<Image, Image> outG = imgR.to_gpu(gpu).inverseRotationWithVingette(Interval(gamma - delta, gamma + delta), isEmptyG, refinements, toInt, refineToInt);
            Image imgRinvG = get<0>(outG).to_cpu();
            Image vingetteG = get<1>(outG).to_cpu();
            REQUIRE( !isEmpty );
            REQUIRE( !isEmptyG );
            if(!imgRinv.equals(imgRinvG, 1e-5)) {
              cout << img << endl;
              cout << gamma << endl;
              cout << imgR << endl;
              cout << imgRinv << endl;
              cout << imgRinvG << endl;
            }
            REQUIRE( imgRinv.equals(imgRinvG, 1e-5) );
            REQUIRE( vingette.equals(vingetteG, 1e-5) );
          }
    }
    img = img.resize(32, 32, false);
    img.randomize(random_generator);
    gamma = distribution(random_generator);
  }  
}



TEST_CASE( "addition behaves as expected", "[add]" ) {
  Image a(2, 2, 1, "x, 0.25, 0, 1, 0.1");
  Image b(2, 2, 1, "x, 0.25, 0, 1, 0.22");
  Image res(2, 2, 1, "x, 0.5, 0, 2, 0.32");
  REQUIRE( (a + b).equals(res, 1e-5) );
}

TEST_CASE( "addition behaves the same on CPU and GPU", "[add]" ) {
  Image a(2, 2, 1, "x, 0.25, 0, 1, 0.1");
  Image b(2, 2, 1, "x, 0.25, 0, 1, 0.22");
  Image resCPU = a + b;
  int numGPUs = 0;
  for(int gpu = 0; gpu < numGPUs; ++gpu) {
    Image resGPU = (a.to_gpu(gpu) + b.to_gpu(gpu)).to_cpu();
    REQUIRE( resGPU.equals(resCPU, 1e-5) );
  }  
}


TEST_CASE( "subtraction behaves as expected", "[sub]" ) {
  Image a(2, 2, 1, "x, 0.25, 0, 1, 0.22");
  Image b(2, 2, 1, "x, 0.25, 0, 1, 0.1");
  Image res(2, 2, 1, "x, 0, 0, 0, 0.12");
  REQUIRE( (a - b).equals(res, 1e-5) );
}

TEST_CASE( "subtraction behaves the same on CPU and GPU", "[sub]" ) {
  Image a(2, 2, 1, "x, 0.25, 0, 1, 0.1");
  Image b(2, 2, 1, "x, 0.25, 0, 1, 0.22");
  Image resCPU = a - b;
  int numGPUs = 0;
  for(int gpu = 0; gpu < numGPUs; ++gpu) {
    Image resGPU = (a.to_gpu(gpu) - b.to_gpu(gpu)).to_cpu();
    REQUIRE( resGPU.equals(resCPU, 1e-5) );
  }  
}


// l2norm
// l2normC
// l2normDiff
// filter_vignetting
// roundToInt
// split_color_channels
// resize
// filterVignettingL2diffC
// Center_crop
// inverseTranslation
// translate

TEST_CASE( "rotation behaves the same on CPU and GPU", "[rot]" ) {
  random_generator.seed(seed);
  std::uniform_real_distribution<float> distribution(-1,1);

  int numGPUs = 0;
  cudaGetDeviceCount(&numGPUs);
  
  Image img(2, 2, 1, "x, 0, 0, 1, 0");
  float gamma = 0.1;

  for(size_t k = 0; k < random_repeats; ++k) {
    for(int gpu = 0; gpu < numGPUs; ++gpu) {
      Image imgRCPU_noInt = img.rotate(gamma, false);
      Image imgRGPU_noInt = img.to_gpu(gpu).rotate(gamma, false).to_cpu();
      

      if (!imgRCPU_noInt.equals(imgRGPU_noInt, 1e-5)) {
        cout << imgRCPU_noInt << endl;
        cout << imgRGPU_noInt << endl;
      }
      REQUIRE(imgRCPU_noInt.equals(imgRGPU_noInt, 1e-5));
      
      Image imgRCPU_Int = img.rotate(gamma, true);
      Image imgRGPU_Int = img.to_gpu(gpu).rotate(gamma, true).to_cpu();

      if (!imgRCPU_Int.equals(imgRGPU_Int, 1e-5)) {
        cout << imgRCPU_Int << endl;
        cout << imgRGPU_Int << endl;
      }

      REQUIRE(imgRCPU_Int.equals(imgRGPU_Int, 1e-5));
    }
    img.randomize(random_generator);
    gamma = distribution(random_generator);    
  }
}

TEST_CASE( "putting the image on gpu does not change it", "[gpu]" ) {
  random_generator.seed(seed);
  int numGPUs = 0;
  cudaGetDeviceCount(&numGPUs);
  Image img(2, 2, 1, "x, 0, 0, 1, 0");
  for(size_t k = 0; k < random_repeats; ++k) {
    for(int gpu = 0; gpu < numGPUs; ++gpu) {
      Image img_ = img.to_gpu(gpu).to_cpu();
      REQUIRE(img.equals(img_, 0));
    }
    img.randomize(random_generator);
  }
}


TEST_CASE( "l2 norm works as expected", "[l2norm]" ) {
  Image img1(2, 2, 1, "x, 0, 0, 1, 0");
  REQUIRE(abs(img1.l2norm() - 1) < 1e-5);

  Image img2(2, 2, 1, "x, 0, 1, 1, 0");
  REQUIRE(abs(img2.l2norm() - sqrt(2)) < 1e-5);
}

TEST_CASE( "l2 norm the same on cpu and gpu", "[l2norm gpu]" ) {
  random_generator.seed(seed);
  int numGPUs = 0;
  cudaGetDeviceCount(&numGPUs);
  Image img(15, 15, 1); //GPU expects minimum size 128
  img.randomize(random_generator);
  for(size_t k = 0; k < random_repeats; ++k) {
    float nCPU = img.l2norm();
    for(int gpu = 0; gpu < numGPUs; ++gpu) {
      float nGPU = img.to_gpu(gpu).l2norm();
      REQUIRE(abs(nCPU - nGPU) < 1e-4);
    }
    img.randomize(random_generator);
  }
}



TEST_CASE( "channelwise l2 norm works as expected", "[l2normC]" ) {
  Image img1(2, 2, 2, "x, 0, 0, 1, 1, 0, 0, 0, 0");
  vector<float> norm11 = img1.l2normC(1);
  vector<float> norm12 = img1.l2normC(2);
  REQUIRE(norm11.size() == 2);
  REQUIRE(abs(norm11[0] - 1) < 1e-5);
  REQUIRE(abs(norm11[1] - 1) < 1e-5);
  REQUIRE(norm12.size() == 1);
  REQUIRE(abs(norm12[0] - sqrt(2)) < 1e-5);
  REQUIRE(abs(norm12[0] - img1.l2norm()) < 1e-4);
  REQUIRE(abs(sqrt(norm11[0]*norm11[0] + norm11[1]*norm11[1]) - img1.l2norm()) < 1e-5);
  

  Image img2(2, 2, 2, "x, 0, 1, 1, 0, 0, 1, 1, 0");
  vector<float> norm21 = img2.l2normC(1);
  vector<float> norm22 = img2.l2normC(2);
  REQUIRE(norm21.size() == 2);
  REQUIRE(abs(norm21[0] - sqrt(2)) < 1e-5);
  REQUIRE(abs(norm21[1] - sqrt(2)) < 1e-5);
  REQUIRE(norm22.size() == 1);
  REQUIRE(abs(norm22[0] - 2) < 1e-5);
  REQUIRE(abs(norm22[0] - img2.l2norm()) < 1e-4);
  REQUIRE(abs(sqrt(norm21[0]*norm21[0] + norm21[1]*norm21[1]) - img2.l2norm()) < 1e-5);
}

TEST_CASE( "channelwise l2 norm the same on cpu and gpu", "[l2normC gpu]" ) {
  random_generator.seed(seed);
  int numGPUs = 0;
  cudaGetDeviceCount(&numGPUs);
  Image img(15, 15, 2); //GPU expects minimum size 128
  img.randomize(random_generator);
  for(size_t k = 0; k < random_repeats; ++k) {
    for (size_t c : {1, 2}) {
      vector<float> nCPU = img.l2normC(c);
      for(int gpu = 0; gpu < numGPUs; ++gpu) {
        vector<float> nGPU = img.to_gpu(gpu).l2normC(c);
        REQUIRE(nCPU.size() == nGPU.size());
        for(size_t i = 0; i < nCPU.size(); ++i)
          REQUIRE(abs(nCPU[i] - nGPU[i]) < 1e-4);
      }
    }
    img.randomize(random_generator);
  }
}

TEST_CASE( "l2-difference gives the same result as subtraction followed by l2 norm; also cpu = gpu", "[l2diffC gpu]" ) {
  random_generator.seed(seed);
  int numGPUs = 0;
  cudaGetDeviceCount(&numGPUs);
  Image img1(15, 15, 2); //GPU expects minimum size 128
  Image img2(15, 15, 2); //GPU expects minimum size 128
  for(size_t k = 0; k < random_repeats; ++k) {
    img1.randomize(random_generator);
    img2.randomize(random_generator);
        
    for (size_t c : {1, 2}) {
      Image diff_cpu = img1 - img2;
      vector<float> ndiff_cpu = diff_cpu.l2normC(c);
      vector<float> n_cpu = img1.l2diffC(img2, c);
      REQUIRE(ndiff_cpu.size() == n_cpu.size());
      for(size_t i = 0; i < ndiff_cpu.size(); ++i) {
        REQUIRE(abs(ndiff_cpu[i] - n_cpu[i]) < 1e-4);
      }
     
      for(int gpu = 0; gpu < numGPUs; ++gpu) {      
        Image diff_gpu = img1.to_gpu(gpu) - img2.to_gpu(gpu);      
        vector<float> ndiff_gpu = diff_gpu.l2normC(c);
        vector<float> n_gpu = img1.to_gpu(gpu).l2diffC(img2.to_gpu(gpu), c);
        REQUIRE(ndiff_cpu.size() == ndiff_gpu.size());
        REQUIRE(ndiff_cpu.size() == n_gpu.size());
        for(size_t i = 0; i < ndiff_cpu.size(); ++i) {
          REQUIRE(abs(ndiff_cpu[i] - ndiff_gpu[i]) < 1e-4);
          REQUIRE(abs(ndiff_cpu[i] - n_gpu[i]) < 1e-4);
        }
      }
    }
  }
}



TEST_CASE( "filter-l2-difference gives the same result as filter, subtraction, l2 norm; also cpu = gpu", "[filterVignettingL2diff gpu]" ) {
  random_generator.seed(seed);
  int numGPUs = 0;
  cudaGetDeviceCount(&numGPUs);

  
  
  Image img1(15, 15, 2); //GPU expects minimum size 128
  Image img2(15, 15, 2); //GPU expects minimum size 128
  for(size_t k = 0; k < random_repeats; ++k) {
    img1.randomize(random_generator);
    img2.randomize(random_generator);
    for (float radiusDecrease : {-1, 2}) {
      for (float filter_sigma : {-1, 1, 5}) {
        for (size_t filter_size : {3}) {
          for (size_t c : {1, 2}) {
        
            float *filter = 0;
            if (filter_sigma > 0) {
              filter = img1.getFilter(filter_sigma, filter_size);
            }        
            Image diff_cpu = img1.filter_vignetting(filter, filter_size, radiusDecrease) - img2.filter_vignetting(filter, filter_size, radiusDecrease);
            vector<float> ndiff_cpu = diff_cpu.l2normC(c);
            vector<float> n_cpu = img1.filterVignettingL2diffC(img2, filter, filter_size, radiusDecrease, c);
            if (filter != 0) {
              img1.delFilter(filter);
              filter = 0;
            }           
    
            REQUIRE(ndiff_cpu.size() == n_cpu.size());
            for(size_t i = 0; i < ndiff_cpu.size(); ++i) {
              REQUIRE(abs(ndiff_cpu[i] - n_cpu[i]) < 1e-4);
            }
     
            for(int gpu = 0; gpu < numGPUs; ++gpu) {
              Image img1g = img1.to_gpu(gpu);
              Image img2g = img2.to_gpu(gpu);

              if (filter_sigma > 0) {
                filter = img1g.getFilter(filter_sigma, filter_size);
              }        
              Image diff_gpu = img1g.filter_vignetting(filter, filter_size, radiusDecrease) - img2g.filter_vignetting(filter, filter_size, radiusDecrease);
              vector<float> ndiff_gpu = diff_gpu.l2normC(c);
              vector<float> n_gpu = img1g.filterVignettingL2diffC(img2g, filter, filter_size, radiusDecrease, c);
              if (filter != 0) {
                img1g.delFilter(filter);
                filter = 0;
              }           

              REQUIRE(ndiff_cpu.size() == ndiff_gpu.size());
              REQUIRE(ndiff_cpu.size() == n_gpu.size());

              
              
              for(size_t i = 0; i < ndiff_cpu.size(); ++i) {
                REQUIRE(abs(ndiff_gpu[i] - n_gpu[i]) < 1e-4);
                
                REQUIRE(abs(ndiff_cpu[i] - ndiff_gpu[i]) < 1e-4);
                REQUIRE(abs(ndiff_cpu[i] - n_gpu[i]) < 1e-4);
              }
            }
          }
        }
      }
    }
  }
}




// TEST_CASE( "getE should produce the same result on gpu, multiple GPUs and cpu", "[getE rot]" ) {
//   random_generator.seed(seed);
//   int numGPUs = 0;
//   cudaGetDeviceCount(&numGPUs);


//   Image img(2, 2, 1, "x, 0, 0, 1, 0");
//   vector<vector<float>> betas;
//   betas.push_back({0.1});

//   // vector<float> epsilons;
//   // float* filter = img.getFilter(1, 1);
//   // bool isEmpty = calc_rot(true, //inv,
//   //                         false, //doInt,
//   //                         1, //refinements,
//   //                         img,
//   //                         HyperBox({Interval(0, 0.1)}),
//   //                         epsilons,
//   //                         {0.1},
//   //                         false,
//   //                         1,
//   //                         0,
//   //                         0,
//   //                         filter,
//   //                         1,
//   //                         -1);


//   vector<int> gpus = {1};
//   //if (numGPUs > 0) gpus.push_back(1);
//   //if (numGPUs > 1) gpus.push_back(numGPUs);

//   vector<vector<float>> outErr;
//   vector<vector<float>> outMaxParam;
  
//   arg_t args;
//   args.img = img;
//   args.initialSplits = 1;
//   args.transform = Transform::rotation;
//   args.target = 0;
//   args.inv = true;
//   args.params = HyperBox({Interval(0, 0.1)});
//   args.betas = betas;
//   args.resize_postTransform = 0;
//   args.center_crop_postTranform = 0;
//   args.filter_sigma = 1;
//   args.filter_size = 1;
//   args.radiusDecrease = -1;
//   args.debug = false;
//   args.batch_size = 20;
//   args.threads = 1;
//   args.GPU = true;
//   args.stopErr = 0.1;
//   args.refinements = 1;
//   args.doInt = false;
//   args.timeout = 5 * 60;
//   args.computeVingette = true;
//   auto out = getE(args);
//   outErr.push_back(get<0>(out));
//   outMaxParam.push_back(get<1>(out));

//   for (size_t k = 0; k < outMaxParam[0].size(); ++k)
//       for(size_t i = 1; i < outErr.size(); ++i) {

//         REQUIRE(outErr[0].size() == outErr[i].size());
//         for (size_t k = 0; k < outErr[0].size(); ++k) {
//           REQUIRE(abs(outErr[0][k] - outErr[i][k]) < 1e-5);
//         }

//         REQUIRE(outMaxParam[0].size() == outMaxParam[i].size());
//         for (size_t k = 0; k < outMaxParam[0].size(); ++k) {
//            REQUIRE(abs(outMaxParam[0][k] - outMaxParam[i][k]) < 1e-5);
//         }        
//       }

//   }






TEST_CASE( "translation behaves the same on CPU and GPU", "[trans]" ) {
  random_generator.seed(seed);
  std::uniform_real_distribution<float> distribution(-1,1);

  int numGPUs = 0;
  cudaGetDeviceCount(&numGPUs);
  
  Image img(2, 2, 1, "x, 0, 0, 1, 0");
  float dx = 0.1;
  float dy = 0.1;

  for(size_t k = 0; k < random_repeats; ++k) {
    for(int gpu = 0; gpu < numGPUs; ++gpu) {
      Image imgRCPU_noInt = img.translate(dx, dy, false);
      Image imgRGPU_noInt = img.to_gpu(gpu).translate(dx, dy, false).to_cpu();
      

      if (!imgRCPU_noInt.equals(imgRGPU_noInt, 1e-5)) {
        cout << imgRCPU_noInt << endl;
        cout << imgRGPU_noInt << endl;
      }
      REQUIRE(imgRCPU_noInt.equals(imgRGPU_noInt, 1e-5));
      
      Image imgRCPU_Int = img.translate(dx, dy, true);
      Image imgRGPU_Int = img.to_gpu(gpu).translate(dx, dy, true).to_cpu();

      if (!imgRCPU_Int.equals(imgRGPU_Int, 1e-5)) {
        cout << imgRCPU_Int << endl;
        cout << imgRGPU_Int << endl;
      }

      REQUIRE(imgRCPU_Int.equals(imgRGPU_Int, 1e-5));
    }
    img.randomize(random_generator);
    dx = distribution(random_generator);
    dy = distribution(random_generator);
  }
}



TEST_CASE( "rect vingette works as intended", "[vingette]" ) {
  Image img(3, 3, 1, "x, 1, 1, 1, 1, 1, 1, 1, 1, 1");
  Image target(3, 3, 1, "x, 0, 0, 0, 0, 1, 0, 0, 0, 0");
  Image out = img.rect_vignetting(1);
  Image outG = img.to_gpu(0).rect_vignetting(1).to_cpu();

  REQUIRE(out.equals(target, 1e-5));
  REQUIRE(outG.equals(target, 1e-5));
}
