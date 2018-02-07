/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "imageNet.h"

#include "loadImage.h"
#include "cudaFont.h"
#include <leveldb/db.h>

#include <iostream>
#include <fstream>
#include <gflags/gflags.h>

DEFINE_string(model_dir,"","directory that contains the model files");
DEFINE_string(img,"","input image file");
DEFINE_string(output_file,"","output file, otherwise classification output is reported in console");


// main entry point
int main( int argc, char** argv )
{
  google::ParseCommandLineFlags(&argc, &argv, true);

  std::string imgFilename = FLAGS_img;
  std::string modelDir = FLAGS_model_dir;


  // create imageNet
  std::string prototxt_path = modelDir + "models/resnet_50/deploy.prototxt";
  std::string model_path = modelDir + "models/resnet_50/model_iter_70000.caffemodel";
  std::string mean_binary = "";
  std::string class_labels = modelDir + "models/resnet_50/corresp.txt";
  imageNet* net = imageNet::Create(prototxt_path.c_str(), model_path.c_str(),
				   mean_binary.c_str(), class_labels.c_str());

  if( !net )
    {
      std::cerr << "imagenet-console:   failed to initialize imageNet\n";
      return 0;
    }

  // load image from file on disk
  float* imgCPU    = NULL;
  float* imgCUDA   = NULL;
  int    imgWidth  = 0;
  int    imgHeight = 0;

  if (!loadImageRGBA(imgFilename.c_str(), (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight))
    {
      std::cerr << "failed to load image " << imgFilename << std::endl;
      return 0;
    }

  float confidence = 0.0f;

  // classify image
  const int img_class = net->Classify(imgCUDA, imgWidth, imgHeight, &confidence);

  if( img_class < 0 )
    std::cerr << "imagenet-console:  failed to classify " << imgFilename << "(result=" << img_class << ")\n";
  else if (FLAGS_output_file.empty())
    {
      std::cerr << "imagenet-console: " << imgFilename << " -> " << confidence * 100.0f << " class #" << img_class << " (" << net->GetClassDesc(img_class) << ")\n";
    }
  else
    {
      std::ofstream of(FLAGS_output_file);
      if (of)
	of << "imagenet-console: " << imgFilename << " -> " << confidence * 100.0f << " class #" << img_class << " (" << net->GetClassDesc(img_class) << ")\n";
      else
	{
	  std::cerr << "failed to write to output file " << FLAGS_output_file << std::endl;
	}
    }

  CUDA(cudaFreeHost(imgCPU));
  delete net;
  return 0;
}
