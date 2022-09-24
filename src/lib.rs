//! A template for creating Rust open-source repo on GitHub
//!
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

use fmmap::{MmapFileExt, MmapFileMut, MmapFileMutExt};
use rand::{distributions::Alphanumeric, thread_rng, Rng};
use std::{
    borrow::Borrow,
    cell::RefCell,
    path::{Path, PathBuf},
    sync::atomic::{AtomicI64, Ordering},
};

const TEMP_FILE_NAME_LEN: usize = 16;
const TEMP_FILE_SUFFIX: &str = "buffer";

///
#[derive(Debug, thiserror::Error)]
pub enum Error {
    ///
    #[error("cannot grow because of exceeds the max buffer size {0}")]
    Oversize(usize),
    ///
    #[error("offset: {0} beyond current size {1}")]
    Overflow(usize, usize),
    ///
    #[error("fail to truncate buffer: {0}")]
    TruncateError(fmmap::error::Error),
    ///
    #[error("fail to mmap temp file")]
    FailToMmapTempFile(#[from] fmmap::error::Error),
    ///
    #[error("IO error")]
    IO(#[from] std::io::Error),

    ///
    #[error("sort buffer start with zero")]
    SortStartZero,
}

/// Config auto mmap
pub struct AutoMmapMeta {
    /// Memory mode falls back to an mmaped tmpfile after crossing this size
    after: usize,
    /// directory for autoMmap to create a tempfile in
    dir: PathBuf,
}

/// Buffer type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferType {
    /// Buffer in memory
    Memory,
    /// Buffer in by memory map to a disk file
    Mmap,
}

/// Buffer is equivalent of Go's bytes.Buffer without the ability to read. It is NOT thread-safe.
///
/// In Memory mode, default allocator is used to allocate memory, which depending upon how the code is
/// compiled could use jemalloc for allocations.
///
/// In Mmap mode, Buffer  uses file mmap to allocate memory. This allows us to store big data
/// structures without using physical memory.
///
/// MaxSize can be set to limit the memory usage.
pub struct Buffer {
    /// number of starting bytes used for padding
    padding: i64,

    /// used length of the buffer
    offset: AtomicI64,

    /// type of the underlying buffer
    buf_type: BufferType,

    /// capacity of the buffer
    cur_size: usize,

    /// causes a panic if the buffer grows beyond this size
    max_size: Option<usize>,

    /// optional mmap backing for the buffer
    mmap_file: MmapFileMut,

    auto_mmap_meta: Option<AutoMmapMeta>,

    /// when enabled, Release will not delete the underlying mmap file
    persistent: bool,

    /// used for jemalloc stats
    tag: &'static str,
}

impl Buffer {
    /// default capacity of the buffer
    pub const DEFAULT_CAPACITY: usize = 64;

    const DEFAULT_TAG: &'static str = "buffer";

    ///
    pub fn new(capacity: usize, mut tag: &'static str) -> Self {
        let capacity = capacity.max(Self::DEFAULT_CAPACITY);
        if tag.is_empty() {
            tag = Self::DEFAULT_TAG;
        }

        let mut mmap_file = MmapFileMut::memory_with_capacity("", capacity);
        mmap_file.truncate(capacity as u64).unwrap();
        Self {
            padding: 8,
            offset: AtomicI64::new(8),
            buf_type: BufferType::Memory,
            cur_size: capacity,
            max_size: None,
            mmap_file,
            auto_mmap_meta: None,
            persistent: false,
            tag,
        }
    }

    ///
    pub fn persistent<P: AsRef<Path>>(capacity: usize, file: P) -> Result<Self, Error> {
        let mut mmap_file = MmapFileMut::open(file).map_err(Error::FailToMmapTempFile)?;
        mmap_file
            .truncate(capacity as u64)
            .map_err(Error::FailToMmapTempFile)?;

        Ok(Self {
            padding: 8,
            offset: AtomicI64::new(8),
            buf_type: BufferType::Mmap,
            cur_size: mmap_file.len(),
            max_size: None,
            mmap_file,
            auto_mmap_meta: None,
            persistent: true,
            tag: "",
        })
    }

    ///
    pub fn temp<P: AsRef<Path>>(capacity: usize, dir: Option<P>) -> Result<Self, Error> {
        let path = if let Some(path) = dir {
            path.as_ref().to_path_buf()
        } else {
            std::env::temp_dir()
        };

        Self::new_buffer_file(path, capacity).map(|mut b| {
            b.persistent = false;
            b
        })
    }

    fn new_buffer_file(path: PathBuf, mut capacity: usize) -> Result<Buffer, Error> {
        if capacity < Self::DEFAULT_CAPACITY {
            capacity = Self::DEFAULT_CAPACITY;
        }

        let mut mmap_file =
            MmapFileMut::create(temp_file(path)).map_err(Error::FailToMmapTempFile)?;

        mmap_file
            .truncate(capacity as u64)
            .map_err(Error::FailToMmapTempFile)?;

        Ok(Self {
            padding: 8,
            offset: AtomicI64::new(8),
            buf_type: BufferType::Mmap,
            cur_size: mmap_file.len(),
            max_size: None,
            mmap_file,
            auto_mmap_meta: None,
            persistent: false,
            tag: "",
        })
    }

    ///
    pub fn with_auto_mmap<NP>(mut self, threshold: usize, path: NP) -> Self
    where
        NP: AsRef<Path>,
    {
        if !matches!(self.buf_type, BufferType::Memory) {
            panic!("can only auto mmap with Memory buffer type");
        }

        let meta = AutoMmapMeta {
            after: threshold,
            dir: path.as_ref().to_path_buf(),
        };

        self.auto_mmap_meta = Some(meta);
        self
    }

    ///
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_size = Some(size);
        self
    }

    ///
    pub fn is_empty(&self) -> bool {
        self.offset.load(Ordering::SeqCst) == self.start_offset()
    }

    ///
    pub fn start_offset(&self) -> i64 {
        self.padding
    }

    /// Returns the number of bytes written to the buffer so far
    /// plus the padding at the start of the buffer.
    pub fn len_with_padding(&self) -> usize {
        self.offset.load(Ordering::SeqCst) as usize
    }

    /// Returns the number of bytes written to the buffer so far
    /// (without the padding).
    pub fn len_without_padding(&self) -> usize {
        (self.offset.load(Ordering::SeqCst) - self.padding) as usize
    }

    /// Returns all the written bytes as a slice
    pub fn bytes(&self) -> &[u8] {
        let off = self.offset.load(Ordering::SeqCst) as usize;
        &self.mmap_file.as_slice()[self.padding as usize..off]
    }

    /// Returns all the written bytes as a mutable slice
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        let off = self.offset.load(Ordering::SeqCst) as usize;
        &mut self.mmap_file.as_mut_slice()[self.padding as usize..off]
    }

    ///
    pub fn data(&self, offset: usize) -> Result<&[u8], Error> {
        if offset > self.cur_size {
            return Err(Error::Overflow(offset, self.cur_size));
        }
        Ok(self.mmap_file.slice(offset, self.cur_size - offset))
    }

    ///
    pub fn data_mut(&mut self, offset: usize) -> Result<&mut [u8], Error> {
        if offset > self.cur_size {
            return Err(Error::Overflow(offset, self.cur_size));
        }
        Ok(self.mmap_file.slice_mut(offset, self.cur_size - offset))
    }

    /// Grow would grow the buffer to have at least n more bytes. In case the buffer is at capacity, it
    /// would reallocate twice the size of current capacity + n, to ensure n bytes can be written to the
    /// buffer without further allocation. In UseMmap mode, this might result in underlying file
    /// expansion.
    pub fn grow(&mut self, n: usize) -> Result<(), Error> {
        let size = (self.offset.load(Ordering::Relaxed) as usize) + n;
        if let Some(max_size) = self.max_size {
            if max_size > 0 && size > max_size {
                return Err(Error::Oversize(max_size));
            }
        }

        if size < self.cur_size {
            return Ok(());
        }

        self.cur_size += (self.cur_size + n) // Calculate new capacity.
            .min(1 << 30) // Don't allocate more than 1GB at a time.
            .max(n); // Allocate at least n, even if it exceeds the 1GB limit above.

        match self.buf_type {
            BufferType::Memory => {
                // If autoMmap gets triggered, copy the slice over to an mmaped file.
                if let Some(ref meta) = self.auto_mmap_meta {
                    if meta.after > 0 && self.cur_size > meta.after {
                        self.buf_type = BufferType::Mmap;
                        let mut file = meta.dir.clone();

                        file.push(
                            thread_rng()
                                .sample_iter(&Alphanumeric)
                                .take(TEMP_FILE_NAME_LEN)
                                .map(char::from)
                                .collect::<String>(),
                        );

                        file.set_extension(TEMP_FILE_SUFFIX);

                        let mut mmap_file =
                            MmapFileMut::create(file).map_err(Error::FailToMmapTempFile)?;

                        if !self.persistent {
                            mmap_file.set_remove_on_drop(true);
                        }

                        mmap_file
                            .truncate(self.cur_size as u64)
                            .map_err(Error::FailToMmapTempFile)?;

                        mmap_file
                            .write_all(self.mmap_file.as_slice(), 0)
                            .map_err(Error::FailToMmapTempFile)?;

                        mmap_file.flush().map_err(Error::FailToMmapTempFile)?;

                        self.mmap_file = mmap_file;
                        return Ok(());
                    }
                }

                // else, reallocate the slice
                // unwrap safe here, because we know the mmap file type is memory,
                // in fmmap, the underlying for memory memmap file is BytesMut,
                // so we are safely resize
                self.mmap_file.truncate(self.cur_size as u64).unwrap();
                Ok(())
            }
            BufferType::Mmap => {
                // Truncate and remap the underlying file.
                self.mmap_file
                    .truncate(self.cur_size as u64)
                    .map_err(Error::TruncateError)
            }
        }
    }

    /// `allocate` is a way to get a slice of size n back from the buffer. This slice can be directly
    /// written to.
    ///
    /// # Warning:
    /// Allocate is not thread-safe. The byte slice returned MUST be used before
    /// further calls to Buffer.
    pub fn allocate(&mut self, n: usize) -> Result<&mut [u8], Error> {
        self.grow(n).map(|_| {
            let off = self.offset.fetch_add(n as i64, Ordering::SeqCst);
            self.mmap_file.slice_mut(off as usize, n)
        })
    }

    /// `allocate_offset` works the same way as allocate, but instead of returning a byte slice, it returns
    /// the offset of the allocation.
    pub fn allocate_offset(&mut self, n: usize) -> Result<usize, Error> {
        self.grow(n)
            .map(|_| self.offset.fetch_add(n as i64, Ordering::SeqCst) as usize)
    }

    /// `slice_allocate` would encode the size provided into the buffer, followed by a call to `allocate`,
    /// hence returning the slice of size sz. This can be used to allocate a lot of small buffers into
    /// this big buffer.
    /// Note that `slice_allocate` should NOT be mixed with normal calls to `write`.
    pub fn slice_allocate(&mut self, size: usize) -> Result<&mut [u8], Error> {
        self.grow(4 + size).and_then(|_| {
            self.slice_allocate_in(size)
                .and_then(|_| self.allocate(size))
        })
    }

    fn slice_allocate_in(&mut self, size: usize) -> Result<(), Error> {
        self.allocate(4).map(|buf| {
            buf.copy_from_slice(&(size as u32).to_be_bytes());
        })
    }

    /// `write` would write p bytes to the buffer.
    pub fn write(&mut self, p: &[u8]) -> Result<usize, Error> {
        let len = p.len();
        self.grow(len).map(|_| {
            let n = self.offset.fetch_add(len as i64, Ordering::SeqCst);
            self.mmap_file.slice_mut(n as usize, len).copy_from_slice(p);
            len
        })
    }

    ///
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), Error> {
        self.slice_allocate(bytes.len()).map(|dst| {
            dst.copy_from_slice(bytes);
        })
    }

    /// `reset` would reset the buffer to be reused.
    pub fn reset(&self) {
        self.offset.store(self.start_offset(), Ordering::SeqCst)
    }

    /// `slice` would return the slice written at offset.
    pub fn slice(&self, offset: i64) -> (&[u8], i64) {
        const EMPTY: &[u8] = &[];
        let cur_offset = self.offset.load(Ordering::SeqCst);
        if offset >= cur_offset {
            return (EMPTY, -1);
        }

        let size = u32::from_be_bytes(
            self.mmap_file.as_slice()[offset as usize..offset as usize + 4]
                .try_into()
                .unwrap(),
        );
        let start = offset + 4;
        let mut next = start + size as i64;
        let res = &self.mmap_file.as_slice()[start as usize..next as usize];

        if next >= cur_offset as i64 {
            next = -1;
        }

        (res, next)
    }

    /// `slice_mut` would return the slice written at offset.
    pub fn slice_mut(&mut self, offset: i64) -> (Option<&mut [u8]>, i64) {
        let cur_offset = self.offset.load(Ordering::SeqCst) as i64;
        if offset >= cur_offset {
            return (None, -1);
        }

        let size = u32::from_be_bytes(
            self.mmap_file.as_slice()[offset as usize..offset as usize + 4]
                .try_into()
                .unwrap(),
        );
        let start = offset + 4;
        let mut next = start + size as i64;
        let res = &mut self.mmap_file.as_mut_slice()[start as usize..next as usize];
        if next >= cur_offset {
            next = -1;
        }

        (Some(res), next)
    }

    /// `slice_offsets` is an expensive function. Use sparingly.
    pub fn slice_offsets(&self) -> Vec<i64> {
        let mut offsets = Vec::with_capacity(8);
        let mut next = self.start_offset();
        while next >= 0 {
            offsets.push(next);
            next = self.slice(next).1;
        }
        offsets
    }

    ///
    pub fn slice_iterate<F>(&self, mut f: F) -> Result<(), Error>
    where
        F: FnMut(&[u8]) -> Result<(), Error>,
    {
        if self.is_empty() {
            return Ok(());
        }

        let mut next = self.start_offset();
        while next >= 0 {
            let (slice, nxt) = self.slice(next);
            next = nxt;
            if slice.is_empty() {
                continue;
            }
            f(slice)?;
        }
        Ok(())
    }

    ///
    pub fn slice_iterate_mut<F>(&mut self, mut f: F) -> Result<(), Error>
    where
        F: Fn(&mut [u8]) -> Result<(), Error>,
    {
        if self.is_empty() {
            return Ok(());
        }

        let mut next = self.start_offset();
        while next >= 0 {
            let (slice, nxt) = self.slice_mut(next);
            next = nxt;
            if let Some(s) = slice {
                if s.is_empty() {
                    continue;
                }

                if let Err(e) = f(s) {
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    fn sort_slice<L>(&mut self, less: L) -> Result<(), Error>
    where
        L: Copy + Fn(&[u8], &[u8]) -> bool,
    {
        self.sort_slice_between(
            self.start_offset(),
            self.offset.load(Ordering::SeqCst),
            less,
        )
    }

    fn sort_slice_between<L>(&mut self, start: i64, end: i64, less: L) -> Result<(), Error>
    where
        L: Copy + Fn(&[u8], &[u8]) -> bool,
    {
        if start >= end {
            return Ok(());
        }

        if start == 0 {
            return Err(Error::SortStartZero);
        }

        let mut offsets = Vec::with_capacity(16);
        let (mut next, mut count) = (start, 0);
        while next >= 0 && next < end {
            if count % 1024 == 0 {
                offsets.push(next);
            }
            let (_, nxt) = self.slice(next);
            next = nxt;
            count += 1;
        }

        assert!(!offsets.is_empty());
        if offsets[offsets.len() - 1] != end {
            offsets.push(end);
        }

        let sz_tmp = (((end - start) / 2) as f64 * 1.1) as usize;
        let tag = self.tag;
        let mut s = SortHelper {
            offsets: offsets.as_slice(),
            original: self,
            tmp: RefCell::new(Buffer::new(sz_tmp, tag)),
            small: {
                let mut vec = Vec::with_capacity(1024);
                vec.fill(0);
                vec
            },
        };

        let mut left = offsets[0];
        for off in offsets.iter().skip(1) {
            s.sort_small(left, *off, less)?;
            left = *off;
        }

        s.sort(0, offsets.len() - 1, less);
        Ok(())
    }
}

struct SortHelper<'a> {
    offsets: &'a [i64],
    original: &'a mut Buffer,
    tmp: RefCell<Buffer>,
    small: Vec<i64>,
}

impl<'a> SortHelper<'a> {
    fn sort_small<L>(&mut self, start: i64, end: i64, mut less: L) -> Result<(), Error>
    where
        L: Copy + Fn(&[u8], &[u8]) -> bool,
    {
        self.tmp.borrow().reset();
        self.small.clear();
        let mut next = start;
        while next >= 0 && next < end {
            self.small.push(next);
            let (_, nxt) = self.original.slice(next);
            next = nxt;
        }

        // We are sorting the slices pointed to by s.small offsets, but only moving the offsets around.
        indexsort::sort_slice(&mut self.small, |small, i, j| {
            let (left, _) = self.original.slice(small[i]);
            let (right, _) = self.original.slice(small[j]);
            less(left, right)
        });

        // Now we iterate over the s.small offsets and copy over the slices. The result is now in order.
        for off in self.small.iter() {
            let src = &self.original.mmap_file.as_slice()[*off as usize..];
            self.tmp.borrow_mut().write(raw_slice(src))?;
        }

        self.original
            .mmap_file
            .write_all(self.tmp.borrow().bytes(), start as usize)
            .map_err(From::from)
    }

    fn sort<L>(&self, lo: usize, hi: usize, less: L) -> &[u8]
    where
        L: Copy + Fn(&[u8], &[u8]) -> bool,
    {
        assert!(lo <= hi);
        let mid = lo + (hi - lo) / 2;
        let (loff, hoff) = (self.offsets[lo] as usize, self.offsets[hi] as usize);

        if lo == mid {
            return &self.original.mmap_file.as_slice()[loff..hoff];
        }

        // lo, mid would sort from [offset[lo], offset[mid]) .
        let left = self.sort(lo, mid, less);
        // Typically we'd use mid+1, but here mid represents an offset in the buffer. Each offset
        // contains a thousand entries. So, if we do mid+1, we'd skip over those entries.
        let right = self.sort(mid, hi, less);

        self.merge(left, right, loff, hoff, less);
        &self.original.mmap_file.as_slice()[loff..hoff]
    }

    fn merge<L>(&self, left: &[u8], right: &[u8], mut start: usize, end: usize, less: L)
    where
        L: Copy + Fn(&[u8], &[u8]) -> bool,
    {
        if left.is_empty() || right.is_empty() {
            return;
        }

        let mut tmp = self.tmp.borrow_mut();
        tmp.reset();
        if let Err(e) = tmp.write(left) {
            panic!("{}", e);
        }
        let mut left = tmp.bytes();
        let mut right = right;
        while start < end {
            if left.is_empty() {
                unsafe {
                    let ptr = (self.original.mmap_file.as_slice().as_ptr() as *mut u8).add(start);
                    std::ptr::copy_nonoverlapping(right.as_ptr(), ptr, end - start);
                }
                return;
            }

            if right.is_empty() {
                unsafe {
                    let ptr = (self.original.mmap_file.as_slice().as_ptr() as *mut u8).add(start);
                    std::ptr::copy_nonoverlapping(left.as_ptr(), ptr, end - start);
                }
                return;
            }

            let ls = raw_slice(left);
            let rs = raw_slice(right);

            match less(&ls[4..], &rs[4..]) {
                true => {
                    let src_len = ls.len();
                    unsafe {
                        let ptr =
                            (self.original.mmap_file.as_slice().as_ptr() as *mut u8).add(start);

                        std::ptr::copy_nonoverlapping(ls.as_ptr(), ptr, src_len);
                    }
                    left = &left[src_len..];
                    start += src_len;
                }
                false => {
                    let src_len = rs.len();
                    unsafe {
                        let ptr =
                            (self.original.mmap_file.as_slice().as_ptr() as *mut u8).add(start);

                        std::ptr::copy_nonoverlapping(rs.as_ptr(), ptr, src_len);
                    }
                    right = &right[src_len..];
                    start += src_len;
                }
            }
        }
    }
}

fn raw_slice(buf: &[u8]) -> &[u8] {
    let sz = u32::from_be_bytes(buf[..4].try_into().unwrap());
    &buf[..4 + sz as usize]
}

#[inline]
fn temp_file(dir: impl AsRef<Path>) -> impl AsRef<Path> {
    let mut dir = dir.as_ref().to_path_buf();
    dir.push(
        thread_rng()
            .sample_iter(&Alphanumeric)
            .take(TEMP_FILE_NAME_LEN)
            .map(char::from)
            .collect::<String>(),
    );
    dir.set_extension(TEMP_FILE_SUFFIX);
    dir
}

#[cfg(test)]
mod tests {
    use rand::RngCore;

    use super::*;

    fn new_test_buffers(capacity: usize) -> Vec<Buffer> {
        vec![
            Buffer::new(capacity, "test"),
            Buffer::temp(capacity, Some("")).unwrap(),
        ]
    }

    fn rand_usize() -> usize {
        let mut rng = thread_rng();
        rng.gen_range(1..100)
    }

    #[test]
    fn test_buffer() {
        let mut rng = thread_rng();
        const CAP: usize = 512;
        let buffers = new_test_buffers(CAP);
        for mut buffer in buffers {
            // This is just for verifying result
            let mut bytes_buf = Buffer::new(CAP, "test");

            // Writer small bytes
            let mut small_data = [0u8; 256];
            rng.fill_bytes(&mut small_data);
            let mut big_data = [0u8; 1024];
            rng.fill_bytes(&mut big_data);

            buffer.write(&small_data).unwrap();
            buffer.write(&big_data).unwrap();

            // Write data to bytesBuffer also, just to match result.
            bytes_buf.write(&small_data).unwrap();
            bytes_buf.write(&big_data).unwrap();
            assert_eq!(buffer.bytes(), bytes_buf.bytes());
        }
    }

    #[test]
    fn test_buffer_write() {
        let mut rng = thread_rng();
        const CAP: usize = 32;
        let buffers = new_test_buffers(CAP);
        for mut buffer in buffers {
            let mut data = [0u8; 128];
            rng.fill_bytes(&mut data);

            // This is just for verifying result
            let mut bytes_buf = Buffer::new(CAP, "test");
            let mut end = 32;
            for _ in 0..3 {
                let n = buffer.write(&data[..end]).unwrap();
                assert_eq!(n, end);

                // append to bb also for testing
                bytes_buf.write(&data[..end]).unwrap();
                assert_eq!(buffer.bytes(), bytes_buf.bytes());
                end *= 2;
            }
        }
    }

    #[test]
    fn test_buffer_auto_mmap() {
        let mut buffer =
            Buffer::new(1 << 20, "test").with_auto_mmap(64 << 20, std::env::temp_dir());

        const N: usize = 128 << 10;

        let mut wb = vec![0; 1024];
        let mut rng = thread_rng();
        for _ in 0..N {
            rng.fill(wb.as_mut_slice());
            let b = buffer.slice_allocate(wb.len()).unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(wb.as_slice().as_ptr(), b.as_mut_ptr(), wb.len());
            }
        }

        eprintln!("Buffer size: {}", buffer.len_with_padding());
    }

    #[test]
    fn test_buffer_simple_sort() {
        let buffers = new_test_buffers(1 << 20);
        let mut rng = rand::thread_rng();
        for mut buffer in buffers {
            (0..25600).for_each(|i| {
                let b = buffer.slice_allocate(4).unwrap();
                b.copy_from_slice(u32::to_be_bytes(rng.gen_range(0..25600)).as_slice());
            });

            buffer
                .sort_slice(|ls, rs| {
                    let left = u32::from_be_bytes(ls[..4].try_into().unwrap());
                    let right = u32::from_be_bytes(rs[..4].try_into().unwrap());
                    left < right
                })
                .unwrap();
            let mut last = 0u32;
            let mut i = 0;
            buffer
                .slice_iterate(|v| {
                    let num = u32::from_be_bytes(v.try_into().unwrap());
                    if num < last {
                        eprintln!("num: {}, idx: {} last: {}", num, i, last);
                    }
                    i += 1;
                    assert!(num >= last);
                    last = num;
                    Ok(())
                })
                .unwrap();
        }
    }

    #[test]
    fn test_buffer_slice() {
        const CAP: usize = 32;
        let buffers = new_test_buffers(CAP);
        let mut rng = rand::thread_rng();
        for mut buffer in buffers {
            let count = 10_000;

            let mut exp = (0..count)
                .map(|_| {
                    let size = 1 + rng.gen_range(0..8);
                    let mut test_buf = vec![0; size];
                    rng.fill_bytes(test_buf.as_mut_slice());

                    let new_slice = buffer.slice_allocate(size).unwrap();
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            test_buf.as_slice().as_ptr(),
                            new_slice.as_mut_ptr(),
                            size,
                        );
                    }

                    test_buf
                })
                .collect::<Vec<_>>();

            fn compare(buf: &Buffer, exp: &Vec<Vec<u8>>) {
                let mut i = 0;
                buf.slice_iterate(|v| {
                    // All the slices returned by the buffer should be equal to what we
                    // inserted earlier.
                    assert_eq!(v, exp[i].as_slice());
                    i += 1;
                    Ok(())
                })
                .unwrap();
            }
            compare(&buffer, &exp);

            eprintln!("Sorting using sort.slice");
            indexsort::sort_slice(&mut exp, |data, i, j| {
                data[i].as_slice().cmp(data[j].as_slice()) == core::cmp::Ordering::Less
            });

            eprintln!("Sorting using buffer.sort_slice");
            buffer
                .sort_slice(|a, b| a.cmp(b) == core::cmp::Ordering::Less)
                .unwrap();
            // same order after sort.
            eprintln!("Done sorting");
            compare(&buffer, &exp);
        }
    }

    #[test]
    fn test_buffer_padding() {
        for mut buf in new_test_buffers(1 << 10) {
            eprintln!("using buffer type: {:?}", buf.buf_type);
            let size = rand_usize();

            let write_offset = buf.allocate_offset(size).unwrap();
            assert_eq!(write_offset as i64, buf.start_offset());

            let mut b = vec![0; size];
            let mut rng = thread_rng();
            rng.fill(b.as_mut_slice());

            unsafe {
                std::ptr::copy_nonoverlapping(b.as_ptr(), buf.bytes_mut().as_mut_ptr(), b.len());
            }
            let data = buf.data(buf.start_offset() as usize).unwrap();
            assert!(b.as_slice().eq(&data[..size]));
        }
    }

    #[test]
    fn test_small_buffer() {
        let mut buf = Buffer::new(5, "test");
        // Write something to buffer so sort actually happens.
        buf.write_bytes(b"abc").unwrap();
        // This test fails if the buffer has offset > currSz.
    }

    #[test]
    fn test_sort() {
        let mut vec = vec![1, 10, 5, 100, 20, 50, 11, 60];
        vec.sort_by(|a, b| a.cmp(b));
        eprintln!("{:?}", vec);
    }
}
