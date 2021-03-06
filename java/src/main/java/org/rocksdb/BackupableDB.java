// Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

package org.rocksdb;

import java.util.List;

/**
 * <p>A subclass of RocksDB which supports
 * backup-related operations.</p>
 *
 * @see org.rocksdb.BackupableDBOptions
 */
public class BackupableDB extends RocksDB {
  /**
   * <p>Open a {@code BackupableDB} under the specified path.
   * Note that the backup path should be set properly in the
   * input BackupableDBOptions.</p>
   *
   * @param opt {@link org.rocksdb.Options} to set for the database.
   * @param bopt {@link org.rocksdb.BackupableDBOptions} to use.
   * @param db_path Path to store data to. The path for storing the backup
   *   should be specified in the {@link org.rocksdb.BackupableDBOptions}.
   *
   * @return {@link BackupableDB} reference to the opened database.
   *
   * @throws RocksDBException thrown if error happens in underlying
   *    native library.
   */
  public static BackupableDB open(
      final Options opt, final BackupableDBOptions bopt, final String db_path)
      throws RocksDBException {

    final RocksDB db = RocksDB.open(opt, db_path);
    final BackupableDB bdb = new BackupableDB(open(db.nativeHandle_,
        bopt.nativeHandle_));

    // Prevent the RocksDB object from attempting to delete
    // the underly C++ DB object.
    db.disOwnNativeHandle();

    return bdb;
  }

  /**
   * <p>Captures the state of the database in the latest backup.
   * Note that this function is not thread-safe.</p>
   *
   * @param flushBeforeBackup if true, then all data will be flushed
   *     before creating backup.
   *
   * @throws RocksDBException thrown if error happens in underlying
   *    native library.
   */
  public void createNewBackup(final boolean flushBeforeBackup)
      throws RocksDBException {
    assert(isOwningHandle());
    createNewBackup(nativeHandle_, flushBeforeBackup);
  }

  /**
   * <p>Deletes old backups, keeping latest numBackupsToKeep alive.</p>
   *
   * @param numBackupsToKeep Number of latest backups to keep.
   *
   * @throws RocksDBException thrown if error happens in underlying
   *    native library.
   */
  public void purgeOldBackups(final int numBackupsToKeep)
      throws RocksDBException {
    assert(isOwningHandle());
    purgeOldBackups(nativeHandle_, numBackupsToKeep);
  }

  /**
   * <p>Deletes a specific backup.</p>
   *
   * @param backupId of backup to delete.
   *
   * @throws RocksDBException thrown if error happens in underlying
   *    native library.
   */
  public void deleteBackup(final int backupId) throws RocksDBException {
    assert(isOwningHandle());
    deleteBackup0(nativeHandle_, backupId);
  }

  /**
   * <p>Returns a list of {@link BackupInfo} instances, which describe
   * already made backups.</p>
   *
   * @return List of {@link BackupInfo} instances.
   */
  public List<BackupInfo> getBackupInfos() {
    assert(isOwningHandle());
    return getBackupInfo(nativeHandle_);
  }

  /**
   * <p>Returns a list of corrupted backup ids. If there
   * is no corrupted backup the method will return an
   * empty list.</p>
   *
   * @return array of backup ids as int ids.
   */
  public int[] getCorruptedBackups() {
    assert(isOwningHandle());
    return getCorruptedBackups(nativeHandle_);
  }

  /**
   * <p>Will delete all the files we don't need anymore. It will
   * do the full scan of the files/ directory and delete all the
   * files that are not referenced.</p>
   *
   * @throws RocksDBException thrown if error happens in underlying
   *    native library.
   */
  public void garbageCollect() throws RocksDBException {
    assert(isOwningHandle());
    garbageCollect(nativeHandle_);
  }

  /**
   * <p>Close the BackupableDB instance and release resource.</p>
   *
   * <p>Internally, {@link BackupableDB} owns the {@code rocksdb::DB}
   * pointer to its associated {@link org.rocksdb.RocksDB}.
   * The release of that RocksDB pointer is handled in the destructor
   * of the c++ {@code rocksdb::BackupableDB} and should be transparent
   * to Java developers.</p>
   */
  @Override public void close() {
      super.close();
  }

  /**
   * <p>A protected construction that will be used in the static
   * factory method {@link #open(Options, BackupableDBOptions, String)}.
   * </p>
   *
   * @param nativeHandle The native handle of the C++ BackupableDB object
   */
  protected BackupableDB(final long nativeHandle) {
    super(nativeHandle);
  }

  @Override protected void finalize() throws Throwable {
    close();
    super.finalize();
  }

  protected native static long open(final long rocksDBHandle,
      final long backupDBOptionsHandle);
  protected native void createNewBackup(long handle, boolean flag)
      throws RocksDBException;
  protected native void purgeOldBackups(long handle, int numBackupsToKeep)
      throws RocksDBException;
  private native void deleteBackup0(long nativeHandle, int backupId)
      throws RocksDBException;
  protected native List<BackupInfo> getBackupInfo(long handle);
  private native int[] getCorruptedBackups(long handle);
  private native void garbageCollect(long handle)
      throws RocksDBException;
}
