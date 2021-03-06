// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

package org.rocksdb;

public class SstFileWriter extends RocksObject {
  static {
    RocksDB.loadLibrary();
  }

  public SstFileWriter(final EnvOptions envOptions, final Options options,
      final AbstractComparator<? extends AbstractSlice<?>> comparator) {
    super(newSstFileWriter(
        envOptions.nativeHandle_, options.nativeHandle_, comparator.getNativeHandle()));
  }

  public SstFileWriter(final EnvOptions envOptions, final Options options) {
    super(newSstFileWriter(
        envOptions.nativeHandle_, options.nativeHandle_));
  }

  public void open(final String filePath) throws RocksDBException {
    open(nativeHandle_, filePath);
  }

  public void add(final Slice key, final Slice value) throws RocksDBException {
    add(nativeHandle_, key.getNativeHandle(), value.getNativeHandle());
  }

  public void add(final DirectSlice key, final DirectSlice value) throws RocksDBException {
    add(nativeHandle_, key.getNativeHandle(), value.getNativeHandle());
  }

  public void finish() throws RocksDBException {
    finish(nativeHandle_);
  }

  private native static long newSstFileWriter(
      final long envOptionsHandle, final long optionsHandle, final long userComparatorHandle);

  private native static long newSstFileWriter(final long envOptionsHandle,
      final long optionsHandle);

  private native void open(final long handle, final String filePath) throws RocksDBException;

  private native void add(final long handle, final long keyHandle, final long valueHandle)
      throws RocksDBException;

  private native void finish(final long handle) throws RocksDBException;

  @Override protected final native void disposeInternal(final long handle);
}
