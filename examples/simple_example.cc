// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include <cstdio>
#include <string>
#include <stdlib.h>
#include <sys/time.h>
#include "rocksdb/db.h"
#include "rocksdb/slice.h"
#include "rocksdb/options.h"

using namespace rocksdb;


std::string kDBPath = "/mnt/ssd";

int main() {
  DB* db;
  int ops = 10000000;
  size_t KEY_SIZE = 16;
  size_t VALUE_SIZE = 1024;
  char keybuf[KEY_SIZE + 1];
  char valuebuf[VALUE_SIZE + 1];
  Options options;
  struct timeval begin1,begin2,end1,end2;
  // Optimize RocksDB. This is the easiest way to get RocksDB to perform well
  options.IncreaseParallelism();
  options.OptimizeLevelStyleCompaction();
  // create the DB if it's not already present
  options.create_if_missing = true;
  // open DB
  Status s = DB::Open(options, kDBPath, &db);
  assert(s.ok());



  printf("******Test Start.******\n");
  printf("begin put\n");
  for(uint64_t i = 0; i < ops; i ++) {
    snprintf(keybuf, sizeof(keybuf), "%07d", i);
    snprintf(valuebuf, sizeof(valuebuf), "%020d", i * i);
    std::string data(keybuf, KEY_SIZE);
    std::string value(valuebuf, VALUE_SIZE);
    // Put key-value
    s = db->Put(WriteOptions(), data, value);
    assert(s.ok());
  }
  
  
  // Random get value
  printf("******before Delete; Get begin.******\n");
  gettimeofday(&begin1, NULL);
  for(uint64_t i = 0; i < ops; i ++) {
    std::string get_value;
    uint64_t j = rand()%ops;
    snprintf(keybuf, sizeof(keybuf), "%07d", j);
    snprintf(valuebuf, sizeof(valuebuf), "%020d", i * i);
    std::string data(keybuf, KEY_SIZE);
    std::string value(valuebuf, VALUE_SIZE);
    s = db->Get(ReadOptions(), data, &get_value);
    assert(s.ok());
    assert(get_value == value);
  }
  gettimeofday(&end1, NULL);
  double delta1 = (end1.tv_sec-begin1.tv_sec) + (end1.tv_usec-begin1.tv_usec)/1000000.0;
  printf("******Get finished.******\n");
  printf("end\n Get 总共时间：%lf s\n",delta1);

  
  for(uint64_t i = 0; i < ops; i ++) {
    if(i%10 == 1){
      snprintf(keybuf, sizeof(keybuf), "%07d", i);
      std::string data(keybuf, KEY_SIZE);
      // atomically apply 
      {
        WriteBatch batch;
        batch.Delete(data);
        s = db->Write(WriteOptions(), &batch);
      }
    }
  }

  sleep(100);

  // Random get value
  printf("******after Delete; Get begin.******\n");
  gettimeofday(&begin2, NULL);
  for(uint64_t i = 0; i < ops; i ++) {
    std::string get_value;
    uint64_t j = rand()%ops;
    snprintf(keybuf, sizeof(keybuf), "%07d", j);
    snprintf(valuebuf, sizeof(valuebuf), "%020d", i * i);
    std::string data(keybuf, KEY_SIZE);
    std::string value(valuebuf, VALUE_SIZE);
    s = db->Get(ReadOptions(), data, &get_value);
    assert(s.ok());
    assert(get_value == value);
  }
  gettimeofday(&end2, NULL);
  double delta2 = (end2.tv_sec-begin2.tv_sec) + (end2.tv_usec-begin2.tv_usec)/1000000.0;
  printf("******Get finished.******\n");
  printf("end\n Get 总共时间：%lf s\n",delta2);


  delete db;
  return 0;
}
