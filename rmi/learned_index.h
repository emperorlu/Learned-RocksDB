#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <forward_list>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "monitoring/perf_context_imp.h"
#include "rmi.h"
using namespace std;
using namespace rocksdb;
#if !defined(LEARNED_INDEX_H)
#define LEARNED_INDEX_H

#if !defined(COUT_THIS)
#define COUT_THIS(this) std::cerr << this << std::endl
#endif  // COUT_THIS

struct Predicts {
  learned_addr_t pos;
  learned_addr_t start;
  learned_addr_t end;

  friend std::ostream& operator<<(std::ostream& output, const Predicts& p) {
    output << "(" << p.start << "," << p.pos << "," << p.end << ")";
    return output;
  }
};

template <class Val_T, class Weight_T>
class LearnedRangeIndexSingleKey {
 public:
  LearnedRangeIndexSingleKey(const RMIConfig& rmi_config) : rmi(rmi_config) {}

  LearnedRangeIndexSingleKey(const std::string& stages,const RMIConfig& rmi_config,unsigned key_n)
      : rmi(stages, rmi_config, key_n){}

  LearnedRangeIndexSingleKey(const std::string& stages,const RMIConfig& rmi_config)
      : rmi(stages, rmi_config){}

  LearnedRangeIndexSingleKey(const std::vector<std::string>& first,
                             const RMIConfig& rmi_config)
      : rmi(first, rmi_config) {}

  LearnedRangeIndexSingleKey(const std::vector<std::string>& first,
                             const std::vector<std::string>& second,
                             uint64_t num, uint64_t key_n = 0)
      : rmi(first, second), sorted_array_size(num) {
    rmi.key_n = key_n;
  }

  void reset() { sorted_array.clear(); }

  ~LearnedRangeIndexSingleKey() {
  }

  LearnedRangeIndexSingleKey(const LearnedRangeIndexSingleKey&) = delete;
  LearnedRangeIndexSingleKey(LearnedRangeIndexSingleKey&) = delete;

  void insert(const uint64_t key, const Val_T value) {
    Record record = {.key = key, .value = value};
    sorted_array.push_back(record);
    rmi.insert(static_cast<double>(key),static_cast<double>(value));
  }

  void insert(const uint64_t key, const Val_T value, learned_addr_t addr) {
    Record record = {.key = key, .value = value};
    sorted_array.push_back(record);
    rmi.insert_w_idx(static_cast<double>(key), addr);
  }

  void finish_insert(bool train_first = true) {
    rmi.finish_insert(train_first);
  }

  void finish_train() { rmi.finish_train(); }

  Predicts predict(const double key) {
    Predicts res;
    rmi.predict_pos(key, res.pos, res.start, res.end);

    res.start = std::min(std::max(res.start, static_cast<int64_t>(0)),
                         static_cast<int64_t>(rmi.key_n));
    res.end = std::min(res.end, static_cast<int64_t>(rmi.key_n));
    res.pos = std::max(res.pos, static_cast<int64_t>(0));
    if (!(res.end >= 0 && res.end <= rmi.key_n)) {
      fprintf(stderr, "%ld get res end\n", res.end);
      assert(false);
    }
    return res;
  }

  // inline Predicts predict_w_model(const double& key, const unsigned& model) {
  //   Predicts res;
  //   rmi.predict_pos_w_model(key, model, res.pos, res.start, res.end);
  //   return res;
  // }

  inline LinearRegression& get_lr_model(const unsigned& model) const {
    auto second_stage = reinterpret_cast<LRStage*>(rmi.second_stage);
    return second_stage->models[model];
  }

  int predict_pos(const double key) {
    Predicts res;
    int pos, start, end;
    rmi.predict(key, pos, start, end);
    res.pos = pos;
    return res.pos;
  }

  /*!
    return which keys belong to this model
   */
  int get_model(const double key) {
    // TODO: not implemented
    return rmi.pick_model_for_key(key);
  }

  void printR() {
    cout << "----result-----" << endl;
    double model_size = 0;
    double model_pram = 0;
    for (auto& m : rmi.first_stage->models) {
      // param.push_back(LinearRegression::serialize_hardcore(m));
      // model_size += sizeof(m.max_error);
      // model_size += sizeof(m.min_error);
      model_size += sizeof(m.bias);
      model_size += sizeof(m.w);
      model_pram += sizeof(m.bias);
      model_pram += sizeof(m.w);
    }

    for (auto& m : rmi.second_stage->models) {
      // model_size += sizeof(m.max_error);
      // model_size += sizeof(m.min_error);
      model_size += sizeof(m.bias);
      model_size += sizeof(m.w);
      model_pram += sizeof(m.bias);
      model_pram += sizeof(m.w);
    }
    cout << "model_size: " << model_size << endl;
    cout << "model_pram_size: " << model_pram << endl;
  }

  void serialize(string& param) { 
    
    for (auto& m : rmi.first_stage->models) {
      param.append(LinearRegression::serialize_hardcore(m));
    }
    for (auto& m : rmi.second_stage->models) {
      param.append(LinearRegression::serialize_hardcore(m));
    }
    // std::cout << "before key_n: " << rmi.key_n << std::endl;
    // std::cout << __func__ << " before  key_n param size:" << param.length() << std::endl;
    // string key_num = std::string(sizeof(rmi.key_n), '0');
    char key_num[sizeof(rmi.key_n)];
    memcpy(key_num, &rmi.key_n, sizeof(rmi.key_n));
    // unsigned key_k = 0;
    // memcpy(&key_k, key_num, sizeof(rmi.key_n));
    // std::cout << "before key_k: " << key_k << " ;sizeof(rmi.key_n): " << sizeof(rmi.key_n) << std::endl;
    param.append(key_num, sizeof(rmi.key_n));
    // std::cout << __func__ << " param size:" << param.length() << std::endl;
  }

  Val_T get(const double key) {
    PERF_TIMER_GUARD(block_seek_nanos);
    learned_addr_t value;
    rmi.predict_pos(key, value);
    return value;
  }

 private:
  struct Trail {
    Val_T val;
    Trail* next;
  };

  struct Head {
    double key = 0.0;
    union {
      Val_T val;
      Trail* next;
    } val_or_ptr;
    bool is_val = false;
  };

  struct Record {
    uint64_t key;
    Val_T value;
  };

 public:
  uint64_t sorted_array_size = 0;

  RMINew<Weight_T> rmi;
  std::vector<Record> sorted_array;
};

#endif  // LEARNED_INDEX_H
