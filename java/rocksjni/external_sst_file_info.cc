// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).
//
// This file implements the "bridge" between Java and C++ and enables
// calling c++ rocksdb::BackupableDB and rocksdb::BackupableDBOptions methods
// from Java side.

#include <jni.h>

#include "include/org_rocksdb_ExternalSstFileInfo.h"
#include "rocksdb/sst_file_writer.h"

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    newExternalSstFileInfo
 * Signature: ()J
 */
jlong Java_org_rocksdb_ExternalSstFileInfo_newExternalSstFileInfo__(
    JNIEnv *env, jclass jcls) {
  return reinterpret_cast<jlong>(new rocksdb::ExternalSstFileInfo());
}

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    newExternalSstFileInfo
 * Signature: (Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;JJII)J
 */
jlong Java_org_rocksdb_ExternalSstFileInfo_newExternalSstFileInfo__Ljava_lang_String_2Ljava_lang_String_2Ljava_lang_String_2JJII(
    JNIEnv *env, jclass jcls, jstring jfile_path, jstring jsmallest_key,
    jstring jlargest_key, jlong jsequence_number, jlong jfile_size,
    jint jnum_entries, jint jversion) {
  const char *file_path = env->GetStringUTFChars(jfile_path, nullptr);
  if(file_path == nullptr) {
    // exception thrown: OutOfMemoryError
    return 0;
  }
  const char *smallest_key = env->GetStringUTFChars(jsmallest_key, nullptr);
  if(smallest_key == nullptr) {
    // exception thrown: OutOfMemoryError
    env->ReleaseStringUTFChars(jfile_path, file_path);
    return 0;
  }
  const char *largest_key = env->GetStringUTFChars(jlargest_key, nullptr);
  if(largest_key == nullptr) {
    // exception thrown: OutOfMemoryError
    env->ReleaseStringUTFChars(jsmallest_key, smallest_key);
    env->ReleaseStringUTFChars(jfile_path, file_path);
    return 0;
  }

  auto *external_sst_file_info = new rocksdb::ExternalSstFileInfo(
      file_path, smallest_key, largest_key,
      static_cast<rocksdb::SequenceNumber>(jsequence_number),
      static_cast<uint64_t>(jfile_size), static_cast<int32_t>(jnum_entries),
      static_cast<int32_t>(jversion));

  env->ReleaseStringUTFChars(jlargest_key, largest_key);
  env->ReleaseStringUTFChars(jsmallest_key, smallest_key);
  env->ReleaseStringUTFChars(jfile_path, file_path);

  return reinterpret_cast<jlong>(external_sst_file_info);
}

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    setFilePath
 * Signature: (JLjava/lang/String;)V
 */
void Java_org_rocksdb_ExternalSstFileInfo_setFilePath(JNIEnv *env, jobject jobj,
                                                      jlong jhandle,
                                                      jstring jfile_path) {
  auto *external_sst_file_info =
      reinterpret_cast<rocksdb::ExternalSstFileInfo *>(jhandle);
  const char *file_path = env->GetStringUTFChars(jfile_path, nullptr);
  if(file_path == nullptr) {
    // exception thrown: OutOfMemoryError
    return;
  }
  external_sst_file_info->file_path = file_path;
  env->ReleaseStringUTFChars(jfile_path, file_path);
}

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    filePath
 * Signature: (J)Ljava/lang/String;
 */
jstring Java_org_rocksdb_ExternalSstFileInfo_filePath(JNIEnv *env, jobject jobj,
                                                      jlong jhandle) {
  auto *external_sst_file_info =
      reinterpret_cast<rocksdb::ExternalSstFileInfo *>(jhandle);
  return env->NewStringUTF(external_sst_file_info->file_path.data());
}

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    setSmallestKey
 * Signature: (JLjava/lang/String;)V
 */
void Java_org_rocksdb_ExternalSstFileInfo_setSmallestKey(
    JNIEnv *env, jobject jobj, jlong jhandle, jstring jsmallest_key) {
  auto *external_sst_file_info =
      reinterpret_cast<rocksdb::ExternalSstFileInfo *>(jhandle);
  const char *smallest_key = env->GetStringUTFChars(jsmallest_key, nullptr);
  if(smallest_key == nullptr) {
    // exception thrown: OutOfMemoryError
    return;
  }
  external_sst_file_info->smallest_key = smallest_key;
  env->ReleaseStringUTFChars(jsmallest_key, smallest_key);
}

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    smallestKey
 * Signature: (J)Ljava/lang/String;
 */
jstring Java_org_rocksdb_ExternalSstFileInfo_smallestKey(JNIEnv *env,
                                                         jobject jobj,
                                                         jlong jhandle) {
  auto *external_sst_file_info =
      reinterpret_cast<rocksdb::ExternalSstFileInfo *>(jhandle);
  return env->NewStringUTF(external_sst_file_info->smallest_key.data());
}

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    setLargestKey
 * Signature: (JLjava/lang/String;)V
 */
void Java_org_rocksdb_ExternalSstFileInfo_setLargestKey(JNIEnv *env,
                                                        jobject jobj,
                                                        jlong jhandle,
                                                        jstring jlargest_key) {
  auto *external_sst_file_info =
      reinterpret_cast<rocksdb::ExternalSstFileInfo *>(jhandle);
  const char *largest_key = env->GetStringUTFChars(jlargest_key, NULL);
  if(largest_key == nullptr) {
    // exception thrown: OutOfMemoryError
    return;
  }
  external_sst_file_info->largest_key = largest_key;
  env->ReleaseStringUTFChars(jlargest_key, largest_key);
}

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    largestKey
 * Signature: (J)Ljava/lang/String;
 */
jstring Java_org_rocksdb_ExternalSstFileInfo_largestKey(JNIEnv *env,
                                                        jobject jobj,
                                                        jlong jhandle) {
  auto *external_sst_file_info =
      reinterpret_cast<rocksdb::ExternalSstFileInfo *>(jhandle);
  return env->NewStringUTF(external_sst_file_info->largest_key.data());
}

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    setSequenceNumber
 * Signature: (JJ)V
 */
void Java_org_rocksdb_ExternalSstFileInfo_setSequenceNumber(
    JNIEnv *env, jobject jobj, jlong jhandle, jlong jsequence_number) {
  auto *external_sst_file_info =
      reinterpret_cast<rocksdb::ExternalSstFileInfo *>(jhandle);
  external_sst_file_info->sequence_number =
      static_cast<rocksdb::SequenceNumber>(jsequence_number);
}

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    sequenceNumber
 * Signature: (J)J
 */
jlong Java_org_rocksdb_ExternalSstFileInfo_sequenceNumber(JNIEnv *env,
                                                          jobject jobj,
                                                          jlong jhandle) {
  auto *external_sst_file_info =
      reinterpret_cast<rocksdb::ExternalSstFileInfo *>(jhandle);
  return static_cast<jlong>(external_sst_file_info->sequence_number);
}

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    setFileSize
 * Signature: (JJ)V
 */
void Java_org_rocksdb_ExternalSstFileInfo_setFileSize(JNIEnv *env, jobject jobj,
                                                      jlong jhandle,
                                                      jlong jfile_size) {
  auto *external_sst_file_info =
      reinterpret_cast<rocksdb::ExternalSstFileInfo *>(jhandle);
  external_sst_file_info->file_size = static_cast<uint64_t>(jfile_size);
}

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    fileSize
 * Signature: (J)J
 */
jlong Java_org_rocksdb_ExternalSstFileInfo_fileSize(JNIEnv *env, jobject jobj,
                                                    jlong jhandle) {
  auto *external_sst_file_info =
      reinterpret_cast<rocksdb::ExternalSstFileInfo *>(jhandle);
  return static_cast<jlong>(external_sst_file_info->file_size);
}

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    setNumEntries
 * Signature: (JI)V
 */
void Java_org_rocksdb_ExternalSstFileInfo_setNumEntries(JNIEnv *env,
                                                        jobject jobj,
                                                        jlong jhandle,
                                                        jint jnum_entries) {
  auto *external_sst_file_info =
      reinterpret_cast<rocksdb::ExternalSstFileInfo *>(jhandle);
  external_sst_file_info->num_entries = static_cast<uint64_t>(jnum_entries);
}

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    numEntries
 * Signature: (J)I
 */
jint Java_org_rocksdb_ExternalSstFileInfo_numEntries(JNIEnv *env, jobject jobj,
                                                     jlong jhandle) {
  auto *external_sst_file_info =
      reinterpret_cast<rocksdb::ExternalSstFileInfo *>(jhandle);
  return static_cast<jint>(external_sst_file_info->num_entries);
}

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    setVersion
 * Signature: (JI)V
 */
void Java_org_rocksdb_ExternalSstFileInfo_setVersion(JNIEnv *env, jobject jobj,
                                                     jlong jhandle,
                                                     jint jversion) {
  auto *external_sst_file_info =
      reinterpret_cast<rocksdb::ExternalSstFileInfo *>(jhandle);
  external_sst_file_info->version = static_cast<int32_t>(jversion);
}

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    version
 * Signature: (J)I
 */
jint Java_org_rocksdb_ExternalSstFileInfo_version(JNIEnv *env, jobject jobj,
                                                  jlong jhandle) {
  auto *external_sst_file_info =
      reinterpret_cast<rocksdb::ExternalSstFileInfo *>(jhandle);
  return static_cast<jint>(external_sst_file_info->version);
}

/*
 * Class:     org_rocksdb_ExternalSstFileInfo
 * Method:    disposeInternal
 * Signature: (J)V
 */
void Java_org_rocksdb_ExternalSstFileInfo_disposeInternal(JNIEnv *env,
                                                          jobject jobj,
                                                          jlong jhandle) {
  auto* esfi = reinterpret_cast<rocksdb::ExternalSstFileInfo *>(jhandle);
  assert(esfi != nullptr);
  delete esfi;
}
