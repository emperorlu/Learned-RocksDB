#include "aep_system.h"
#include "nvm_common2.h"
#include <sys/time.h>
#include <cstdlib>
using namespace rocksdb;
// #define USE_MUIL_THREAD


int main(int argc, char **argv)
{
    int to_cache = atoi(argv[1]);
    int num_size = atoi(argv[2]);
    int ioaf = atoi(argv[3]);
    // int is_thread = atoi(argv[2]);
    cout << "begin" << endl;
    struct timeval begin1,begin2,end1,end2;
    rocksdb::aepsystem *db_;
    size_t KEY_SIZE = rocksdb::NVM_KeySize;
    size_t VALUE_SIZE = rocksdb::NVM_ValueSize;
    int i;
    int ops = 1000000 * num_size;
    db_ = new rocksdb::aepsystem;
    db_->is_cache = to_cache;
    int insert_ops = ops / 10;
    int get_ops = ops / 10 * 19;
    if(ioaf == 1){
        insert_ops = ops / 2 * 3;
        get_ops = ops / 2;
    }
    if(ioaf == 2){
        insert_ops = ops;
        get_ops = ops;
    }
    db_->num_size = insert_ops;
    db_->Initialize();
    
    char keybuf[KEY_SIZE + 1];
    char valuebuf[VALUE_SIZE + 1];
    printf("******Test Start.******\n");
    printf("insert_ops:%d ; get_ops:%d\n",insert_ops,get_ops);
    gettimeofday(&begin1, NULL);
    int thread_num = 20;
#ifdef USE_MUIL_THREAD
    vector<future<void>> futurei;
    for(int tid = 0; tid < thread_num; tid ++) {
        uint64_t from = (insert_ops / thread_num) * tid;
        uint64_t to = (tid == thread_num - 1) ? insert_ops : from + (insert_ops / thread_num);
        futurei.push_back(move(async(launch::async,[&db_](int tid, uint64_t from, uint64_t to) {
            size_t KEY_SIZE = rocksdb::NVM_KeySize;
            size_t VALUE_SIZE = rocksdb::NVM_ValueSize;
            char keybuf[KEY_SIZE + 1];
            char valuebuf[VALUE_SIZE + 1];
            for(uint64_t i = from; i < to; i ++) {
                snprintf(keybuf, sizeof(keybuf), "%07d", i);
                snprintf(valuebuf, sizeof(valuebuf), "%020d", i * i);
                string data(keybuf, KEY_SIZE);
                string value(valuebuf, VALUE_SIZE);
                db_->Insert(data, value);
            }
        }, tid, from, to)));
    }
    for(auto &f : futurei) {
        if(f.valid()) {
            f.get();
        }
    }
#else
    for(uint64_t i = 0; i < ops; i ++) {
        snprintf(keybuf, sizeof(keybuf), "%07d", i);
        snprintf(valuebuf, sizeof(valuebuf), "%020d", i * i);
        string data(keybuf, KEY_SIZE);
        string value(valuebuf, VALUE_SIZE);
        db_->Insert(data, value);
    }
#endif
    gettimeofday(&end1, NULL);
    double delta1 = (end1.tv_sec-begin1.tv_sec) + (end1.tv_usec-begin1.tv_usec)/1000000.0;
    printf("******Insert test finished.******\n");
    gettimeofday(&begin2, NULL);
#ifdef USE_MUIL_THREAD
    vector<future<void>> futures;
    for(int tid = 0; tid < thread_num; tid ++) {
        uint64_t from = (get_ops / thread_num) * tid;
        uint64_t to = (tid == thread_num - 1) ? get_ops : from + (get_ops / thread_num);
        futures.push_back(move(async(launch::async,[&db_](int sun, int tid, uint64_t from, uint64_t to) {
            size_t KEY_SIZE = rocksdb::NVM_KeySize;
            size_t VALUE_SIZE = rocksdb::NVM_ValueSize;
            char keybuf[KEY_SIZE + 1];
            char valuebuf[VALUE_SIZE + 1];
            for(uint64_t i = from; i < to; i ++) {
                int j = rand()%sun;
                snprintf(keybuf, sizeof(keybuf), "%07d", j);
                snprintf(valuebuf, sizeof(valuebuf), "%020d", j * j);
                string data(keybuf, KEY_SIZE);
                string value(valuebuf, VALUE_SIZE);
                string tmp_value = db_->Get(data);
                if(tmp_value.size() == 0) {
                    printf("Error: Get key-value faild.(key:%s)\n", data.c_str());
                } else if(strncmp(value.c_str(), tmp_value.c_str(), VALUE_SIZE) != 0) {
                    printf("Error: Get key-value faild.(Expect:%s, but Get %s)\n", value.c_str(), tmp_value.c_str());
                } 
            }
        }, insert_ops, tid, from, to)));
    }
    for(auto &f : futures) {
        if(f.valid()) {
            f.get();
        }
    }
#else 
    for(uint64_t i = 0; i < ops; i ++) {
        snprintf(keybuf, sizeof(keybuf), "%07d", i);
        snprintf(valuebuf, sizeof(valuebuf), "%020d", i * i);
        string data(keybuf, KEY_SIZE);
        string value(valuebuf, VALUE_SIZE);
        string tmp_value = db_->Get(data);
        if(tmp_value.size() == 0) {
            printf("Error: Get key-value faild.(key:%s)\n", data.c_str());
        } else if(strncmp(value.c_str(), tmp_value.c_str(), VALUE_SIZE) != 0) {
            printf("Error: Get key-value faild.(Expect:%s, but Get %s)\n", value.c_str(), tmp_value.c_str());
        } 
    }
#endif
    gettimeofday(&end2, NULL);
    double delta2 = (end2.tv_sec-begin2.tv_sec) + (end2.tv_usec-begin2.tv_usec)/1000000.0;
    printf("******Get test finished.*****\n");

    db_->End();
    printf("end\n Insert 总共时间：%lf s\n",delta1);
    printf("end\n Get 总共时间：%lf s\n",delta2);
    delete db_;
    printf("******Test Over.******\n");
    return 0; 
}
 
// vector<thread *> threads;
//     for(int tid = 0; tid < thread_num; tid ++) {
//         uint64_t from = (ops / thread_num) * tid;
//         uint64_t to = (tid == thread_num - 1) ? ops : from + (ops / thread_num);
//         threads.push_back(new thread([&db_](int tid, uint64_t from, uint64_t to) {
//             size_t KEY_SIZE = rocksdb::NVM_KeySize;
//             size_t VALUE_SIZE = rocksdb::NVM_ValueSize;
//             char keybuf[KEY_SIZE + 1];
//             char valuebuf[VALUE_SIZE + 1];
//             for(uint64_t i = from; i < to; i ++) {
//                 snprintf(keybuf, sizeof(keybuf), "%07d", i);
//                 snprintf(valuebuf, sizeof(valuebuf), "%020d", i * i);
//                 string data(keybuf, KEY_SIZE);
//                 string value(valuebuf, VALUE_SIZE);
//                 // printf("thread %d run\n", tid);
//                 db_->Insert(data, value);
//             }
//             // printf("thread %d end\n", tid);
//         }, tid, from, to));
//         // futures.push_back(move(f));
//     }
//     int tid = 0;
//     for(auto thread : threads) {
//        thread->join();
//     //    printf("thread %d exit\n", tid ++);
//        delete thread;
//     }