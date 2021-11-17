// #include <gtest/gtest.h>

#include "learned_index.h"
#include <iostream>
// #include "../src/datastream/rocksdb_stream.hpp"
using namespace std;
int main(int argc,char *argv[]){

  RMIConfig rmi_config;
  RMIConfig::StageConfig first, second;

  // XD: which is first, and which is second means?
  // does this means that there are only 2 stages for the model?
  // pps: seems my guess is correct
  first.model_type = RMIConfig::StageConfig::LinearRegression;
  first.model_n = 1;

  second.model_n = atoi(argv[1]);
  second.model_type = RMIConfig::StageConfig::LinearRegression;
  rmi_config.stage_configs.push_back(first);
  rmi_config.stage_configs.push_back(second);

  LearnedRangeIndexSingleKey<uint64_t,float> table(rmi_config);
  // RocksStream<uint64_t,uint64_t> db("./testdb");
  // auto it = db.get_iter();

  int count = 0;
  // for(it->begin();it->valid();it->next()) {
  //   auto key = it->key();
  //   auto value = it->value();
  //   LOG(4) << key;
  //   table.insert(key,value);
  //   count++;
  // }
  srand((unsigned)time(NULL)); 
  vector<double> x;
  vector<double> y;
  // vector<double> z;
  double key = 0;
  double value = 0;
  int size = atoi(argv[2]);
  for (int i = 0; i < size; i++){
    key += rand()%100+1;
    // value += rand()%100+1;

    // cout << "train:" << key << ": " << value << endl;
    table.insert(key,value);
    // count++;
    x.push_back(key);
    y.push_back(value);
  }
  // ASSERT_EQ(count,12);
  table.finish_insert();
  table.finish_train();

  // LOG(4) << "finished insert";

  for (int i = 0; i < size; i++){
    key = x[i];
    value = y[i];
    auto value_get = table.get(key);
    double bit = 1.0* (value-value_get) / value;
    // cout << i << " result: " << value_get << " : " << value << "; error:" << (value-value_get) 
    //       << ";error bit: "<< bit << endl;
  }
  table.printR();
  // it->begin();
  // for(it->begin();it->valid();it->next()) {
  //   auto key = it->key();
  //   auto value = table.get(key);
  //   ASSERT_EQ(value,it->value());
  // }

} // end namespace test
