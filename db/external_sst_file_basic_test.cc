//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include <functional>

#include "db/db_test_util.h"
#include "port/port.h"
#include "port/stack_trace.h"
#include "rocksdb/sst_file_writer.h"
#include "util/testutil.h"

namespace rocksdb {

#ifndef ROCKSDB_LITE
class ExternalSSTFileBasicTest : public DBTestBase {
 public:
  ExternalSSTFileBasicTest() : DBTestBase("/external_sst_file_test") {
    sst_files_dir_ = dbname_ + "/sst_files/";
    DestroyAndRecreateExternalSSTFilesDir();
  }

  void DestroyAndRecreateExternalSSTFilesDir() {
    test::DestroyDir(env_, sst_files_dir_);
    env_->CreateDir(sst_files_dir_);
  }

  Status DeprecatedAddFile(const std::vector<std::string>& files,
                           bool move_files = false,
                           bool skip_snapshot_check = false) {
    IngestExternalFileOptions opts;
    opts.move_files = move_files;
    opts.snapshot_consistency = !skip_snapshot_check;
    opts.allow_global_seqno = false;
    opts.allow_blocking_flush = false;
    return db_->IngestExternalFile(files, opts);
  }

  Status GenerateAndAddExternalFile(
      const Options options, std::vector<int> keys, int file_id,
      std::map<std::string, std::string>* true_data) {
    std::string file_path = sst_files_dir_ + ToString(file_id);
    SstFileWriter sst_file_writer(EnvOptions(), options);

    Status s = sst_file_writer.Open(file_path);
    if (!s.ok()) {
      return s;
    }
    for (int k : keys) {
      std::string key = Key(k);
      std::string value = Key(k) + ToString(file_id);
      s = sst_file_writer.Add(key, value);
      (*true_data)[key] = value;
      if (!s.ok()) {
        sst_file_writer.Finish();
        return s;
      }
    }
    s = sst_file_writer.Finish();

    if (s.ok()) {
      IngestExternalFileOptions ifo;
      ifo.allow_global_seqno = true;
      s = db_->IngestExternalFile({file_path}, ifo);
    }

    return s;
  }

  ~ExternalSSTFileBasicTest() { test::DestroyDir(env_, sst_files_dir_); }

 protected:
  std::string sst_files_dir_;
};

TEST_F(ExternalSSTFileBasicTest, Basic) {
  Options options = CurrentOptions();

  SstFileWriter sst_file_writer(EnvOptions(), options);

  // Current file size should be 0 after sst_file_writer init and before open a
  // file.
  ASSERT_EQ(sst_file_writer.FileSize(), 0);

  // file1.sst (0 => 99)
  std::string file1 = sst_files_dir_ + "file1.sst";
  ASSERT_OK(sst_file_writer.Open(file1));
  for (int k = 0; k < 100; k++) {
    ASSERT_OK(sst_file_writer.Add(Key(k), Key(k) + "_val"));
  }
  ExternalSstFileInfo file1_info;
  Status s = sst_file_writer.Finish(&file1_info);
  ASSERT_TRUE(s.ok()) << s.ToString();

  // Current file size should be non-zero after success write.
  ASSERT_GT(sst_file_writer.FileSize(), 0);

  ASSERT_EQ(file1_info.file_path, file1);
  ASSERT_EQ(file1_info.num_entries, 100);
  ASSERT_EQ(file1_info.smallest_key, Key(0));
  ASSERT_EQ(file1_info.largest_key, Key(99));
  // sst_file_writer already finished, cannot add this value
  s = sst_file_writer.Add(Key(100), "bad_val");
  ASSERT_FALSE(s.ok()) << s.ToString();

  DestroyAndReopen(options);
  // Add file using file path
  s = DeprecatedAddFile({file1});
  ASSERT_TRUE(s.ok()) << s.ToString();
  ASSERT_EQ(db_->GetLatestSequenceNumber(), 0U);
  for (int k = 0; k < 100; k++) {
    ASSERT_EQ(Get(Key(k)), Key(k) + "_val");
  }

  DestroyAndRecreateExternalSSTFilesDir();
}

TEST_F(ExternalSSTFileBasicTest, NoCopy) {
  Options options = CurrentOptions();
  const ImmutableCFOptions ioptions(options);

  SstFileWriter sst_file_writer(EnvOptions(), options);

  // file1.sst (0 => 99)
  std::string file1 = sst_files_dir_ + "file1.sst";
  ASSERT_OK(sst_file_writer.Open(file1));
  for (int k = 0; k < 100; k++) {
    ASSERT_OK(sst_file_writer.Add(Key(k), Key(k) + "_val"));
  }
  ExternalSstFileInfo file1_info;
  Status s = sst_file_writer.Finish(&file1_info);
  ASSERT_TRUE(s.ok()) << s.ToString();
  ASSERT_EQ(file1_info.file_path, file1);
  ASSERT_EQ(file1_info.num_entries, 100);
  ASSERT_EQ(file1_info.smallest_key, Key(0));
  ASSERT_EQ(file1_info.largest_key, Key(99));

  // file2.sst (100 => 299)
  std::string file2 = sst_files_dir_ + "file2.sst";
  ASSERT_OK(sst_file_writer.Open(file2));
  for (int k = 100; k < 300; k++) {
    ASSERT_OK(sst_file_writer.Add(Key(k), Key(k) + "_val"));
  }
  ExternalSstFileInfo file2_info;
  s = sst_file_writer.Finish(&file2_info);
  ASSERT_TRUE(s.ok()) << s.ToString();
  ASSERT_EQ(file2_info.file_path, file2);
  ASSERT_EQ(file2_info.num_entries, 200);
  ASSERT_EQ(file2_info.smallest_key, Key(100));
  ASSERT_EQ(file2_info.largest_key, Key(299));

  // file3.sst (110 => 124) .. overlap with file2.sst
  std::string file3 = sst_files_dir_ + "file3.sst";
  ASSERT_OK(sst_file_writer.Open(file3));
  for (int k = 110; k < 125; k++) {
    ASSERT_OK(sst_file_writer.Add(Key(k), Key(k) + "_val_overlap"));
  }
  ExternalSstFileInfo file3_info;
  s = sst_file_writer.Finish(&file3_info);
  ASSERT_TRUE(s.ok()) << s.ToString();
  ASSERT_EQ(file3_info.file_path, file3);
  ASSERT_EQ(file3_info.num_entries, 15);
  ASSERT_EQ(file3_info.smallest_key, Key(110));
  ASSERT_EQ(file3_info.largest_key, Key(124));
  s = DeprecatedAddFile({file1}, true /* move file */);
  ASSERT_TRUE(s.ok()) << s.ToString();
  ASSERT_EQ(Status::NotFound(), env_->FileExists(file1));

  s = DeprecatedAddFile({file2}, false /* copy file */);
  ASSERT_TRUE(s.ok()) << s.ToString();
  ASSERT_OK(env_->FileExists(file2));

  // This file have overlapping values with the existing data
  s = DeprecatedAddFile({file2}, true /* move file */);
  ASSERT_FALSE(s.ok()) << s.ToString();
  ASSERT_OK(env_->FileExists(file3));

  for (int k = 0; k < 300; k++) {
    ASSERT_EQ(Get(Key(k)), Key(k) + "_val");
  }
}

TEST_F(ExternalSSTFileBasicTest, IngestFileWithGlobalSeqnoPickedSeqno) {
  Options options = CurrentOptions();
  DestroyAndReopen(options);
  std::map<std::string, std::string> true_data;

  int file_id = 1;

  ASSERT_OK(GenerateAndAddExternalFile(options, {1, 2, 3, 4, 5, 6}, file_id++,
                                       &true_data));
  // File dont overwrite any keys, No seqno needed
  ASSERT_EQ(dbfull()->GetLatestSequenceNumber(), 0);

  ASSERT_OK(GenerateAndAddExternalFile(options, {10, 11, 12, 13}, file_id++,
                                       &true_data));
  // File dont overwrite any keys, No seqno needed
  ASSERT_EQ(dbfull()->GetLatestSequenceNumber(), 0);

  ASSERT_OK(
      GenerateAndAddExternalFile(options, {1, 4, 6}, file_id++, &true_data));
  // File overwrite some keys, a seqno will be assigned
  ASSERT_EQ(dbfull()->GetLatestSequenceNumber(), 1);

  ASSERT_OK(
      GenerateAndAddExternalFile(options, {11, 15, 19}, file_id++, &true_data));
  // File overwrite some keys, a seqno will be assigned
  ASSERT_EQ(dbfull()->GetLatestSequenceNumber(), 2);

  ASSERT_OK(
      GenerateAndAddExternalFile(options, {120, 130}, file_id++, &true_data));
  // File dont overwrite any keys, No seqno needed
  ASSERT_EQ(dbfull()->GetLatestSequenceNumber(), 2);

  ASSERT_OK(
      GenerateAndAddExternalFile(options, {1, 130}, file_id++, &true_data));
  // File overwrite some keys, a seqno will be assigned
  ASSERT_EQ(dbfull()->GetLatestSequenceNumber(), 3);

  // Write some keys through normal write path
  for (int i = 0; i < 50; i++) {
    ASSERT_OK(Put(Key(i), "memtable"));
    true_data[Key(i)] = "memtable";
  }
  SequenceNumber last_seqno = dbfull()->GetLatestSequenceNumber();

  ASSERT_OK(
      GenerateAndAddExternalFile(options, {60, 61, 62}, file_id++, &true_data));
  // File dont overwrite any keys, No seqno needed
  ASSERT_EQ(dbfull()->GetLatestSequenceNumber(), last_seqno);

  ASSERT_OK(
      GenerateAndAddExternalFile(options, {40, 41, 42}, file_id++, &true_data));
  // File overwrite some keys, a seqno will be assigned
  ASSERT_EQ(dbfull()->GetLatestSequenceNumber(), last_seqno + 1);

  ASSERT_OK(
      GenerateAndAddExternalFile(options, {20, 30, 40}, file_id++, &true_data));
  // File overwrite some keys, a seqno will be assigned
  ASSERT_EQ(dbfull()->GetLatestSequenceNumber(), last_seqno + 2);

  const Snapshot* snapshot = db_->GetSnapshot();

  // We will need a seqno for the file regardless if the file overwrite
  // keys in the DB or not because we have a snapshot
  ASSERT_OK(
      GenerateAndAddExternalFile(options, {1000, 1002}, file_id++, &true_data));
  // A global seqno will be assigned anyway because of the snapshot
  ASSERT_EQ(dbfull()->GetLatestSequenceNumber(), last_seqno + 3);

  ASSERT_OK(
      GenerateAndAddExternalFile(options, {2000, 3002}, file_id++, &true_data));
  // A global seqno will be assigned anyway because of the snapshot
  ASSERT_EQ(dbfull()->GetLatestSequenceNumber(), last_seqno + 4);

  ASSERT_OK(GenerateAndAddExternalFile(options, {1, 20, 40, 100, 150},
                                       file_id++, &true_data));
  // A global seqno will be assigned anyway because of the snapshot
  ASSERT_EQ(dbfull()->GetLatestSequenceNumber(), last_seqno + 5);

  db_->ReleaseSnapshot(snapshot);

  ASSERT_OK(
      GenerateAndAddExternalFile(options, {5000, 5001}, file_id++, &true_data));
  // No snapshot anymore, no need to assign a seqno
  ASSERT_EQ(dbfull()->GetLatestSequenceNumber(), last_seqno + 5);

  size_t kcnt = 0;
  VerifyDBFromMap(true_data, &kcnt, false);
}

TEST_F(ExternalSSTFileBasicTest, FadviseTrigger) {
  Options options = CurrentOptions();
  const int kNumKeys = 10000;

  size_t total_fadvised_bytes = 0;
  rocksdb::SyncPoint::GetInstance()->SetCallBack(
      "SstFileWriter::InvalidatePageCache", [&](void* arg) {
        size_t fadvise_size = *(reinterpret_cast<size_t*>(arg));
        total_fadvised_bytes += fadvise_size;
      });
  rocksdb::SyncPoint::GetInstance()->EnableProcessing();

  std::unique_ptr<SstFileWriter> sst_file_writer;

  std::string sst_file_path = sst_files_dir_ + "file_fadvise_disable.sst";
  sst_file_writer.reset(
      new SstFileWriter(EnvOptions(), options, nullptr, false));
  ASSERT_OK(sst_file_writer->Open(sst_file_path));
  for (int i = 0; i < kNumKeys; i++) {
    ASSERT_OK(sst_file_writer->Add(Key(i), Key(i)));
  }
  ASSERT_OK(sst_file_writer->Finish());
  // fadvise disabled
  ASSERT_EQ(total_fadvised_bytes, 0);


  sst_file_path = sst_files_dir_ + "file_fadvise_enable.sst";
  sst_file_writer.reset(
      new SstFileWriter(EnvOptions(), options, nullptr, true));
  ASSERT_OK(sst_file_writer->Open(sst_file_path));
  for (int i = 0; i < kNumKeys; i++) {
    ASSERT_OK(sst_file_writer->Add(Key(i), Key(i)));
  }
  ASSERT_OK(sst_file_writer->Finish());
  // fadvise enabled
  ASSERT_EQ(total_fadvised_bytes, sst_file_writer->FileSize());
  ASSERT_GT(total_fadvised_bytes, 0);

  rocksdb::SyncPoint::GetInstance()->DisableProcessing();
}

#endif  // ROCKSDB_LITE

}  // namespace rocksdb

int main(int argc, char** argv) {
  rocksdb::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
