//! A template for creating Rust open-source repo on GitHub
//!
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

use fmmap::{MmapFileExt, MmapFileMut, MmapFileMutExt};
use std::{
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

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
}

/// Config auto mmap
pub struct AutoMmapMeta {
    /// Memory mode falls back to an mmaped tmpfile after crossing this size
    after: usize,
    /// directory for autoMmap to create a tempfile in
    dir: PathBuf,
}

/// Buffer type
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
    padding: u64,

    /// used length of the buffer
    offset: AtomicU64,

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

        Self {
            padding: 8,
            offset: AtomicU64::new(8),
            buf_type: BufferType::Memory,
            cur_size: capacity,
            max_size: None,
            mmap_file: MmapFileMut::memory_with_capacity("", capacity),
            auto_mmap_meta: None,
            persistent: false,
            tag,
        }
    }

    ///
    pub fn with_auto_mmap<NP: AsRef<Path>>(mut self, threshold: usize, path: Option<NP>) -> Self {
        if !matches!(self.buf_type, BufferType::Memory) {
            panic!("can only auto mmap with Memory buffer type");
        }

        let meta = AutoMmapMeta {
            after: threshold,
            dir: path
                .map(|p| p.as_ref().to_path_buf())
                .unwrap_or_else(std::env::temp_dir),
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
    pub fn start_offset(&self) -> u64 {
        self.padding
    }

    /// Returns the number of bytes written to the buffer so far
    /// plus the padding at the start of the buffer.
    pub fn len_with_padding(&self) -> u64 {
        self.offset.load(Ordering::SeqCst)
    }

    /// Returns the number of bytes written to the buffer so far
    /// (without the padding).
    pub fn len_without_padding(&self) -> u64 {
        self.offset.load(Ordering::SeqCst) - self.padding
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
                        // TODO: add more useful methods in fmmap crate
                        // !if self.persistent {
                        // set remove on drop for memmap file
                        // }
                        // let mmap_file = MmapFileMut::create_with_options(path, opts);
                        // self.mmap_file = MmapFileMut::create(
                        //     &meta.dir,
                        //     &format!("{}-{}", self.tag, self.cur_size),
                        //     self.cur_size,
                        // )?;
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
            let off = self.offset.fetch_add(n as u64, Ordering::SeqCst);
            self.mmap_file.slice_mut(off as usize, n)
        })
    }

    /// `allocate_offset` works the same way as allocate, but instead of returning a byte slice, it returns
    /// the offset of the allocation.
    pub fn allocate_offset(&mut self, n: usize) -> Result<usize, Error> {
        self.grow(n)
            .map(|_| self.offset.fetch_add(n as u64, Ordering::SeqCst) as usize)
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
            let n = self.offset.fetch_add(len as u64, Ordering::SeqCst);
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
    pub fn slice(&self, offset: u64) -> Option<(&[u8], u64)> {
        let cur_offset = self.offset.load(Ordering::SeqCst);
        if offset >= cur_offset {
            return None;
        }

        let size = u32::from_be_bytes(
            self.mmap_file.as_slice()[offset as usize..offset as usize + 4]
                .try_into()
                .unwrap(),
        );
        let start = offset + 4;
        let mut next = start + size as u64;
        let res = &self.mmap_file.as_slice()[start as usize..next as usize];

        if next >= cur_offset {
            next -= 1;
        }

        Some((res, next))
    }

    /// `slice_mut` would return the slice written at offset.
    pub fn slice_mut(&mut self, offset: u64) -> Option<(&mut [u8], u64)> {
        let cur_offset = self.offset.load(Ordering::SeqCst);
        if offset >= cur_offset {
            return None;
        }

        let size = u32::from_be_bytes(
            self.mmap_file.as_slice()[offset as usize..offset as usize + 4]
                .try_into()
                .unwrap(),
        );
        let start = offset + 4;
        let mut next = start + size as u64;
        let res = &mut self.mmap_file.as_mut_slice()[start as usize..next as usize];
        if next >= cur_offset {
            next -= 1;
        }

        Some((res, next))
    }

    /// `slice_offsets` is an expensive function. Use sparingly.
    pub fn slice_offsets(&self) -> Vec<u64> {
        let mut offsets = Vec::with_capacity(8);
        let mut next = self.start_offset();
        loop {
            offsets.push(next);
            if let Some((_, n)) = self.slice(next) {
                next = n;
            } else {
                break offsets;
            }
        }
    }

    ///
    pub fn slice_iterate<F>(&self, mut f: F) -> Result<(), Error>
    where
        F: FnMut(&[u8]) -> Result<(), Error>,
    {
        if self.is_empty() {
            return Ok(());
        }

        let mut slice;
        let mut next = self.start_offset();
        loop {
            match self.slice(next) {
                Some((s, n)) => {
                    slice = s;
                    next = n;
                    if slice.is_empty() {
                        continue;
                    }
                    f(slice)?;
                }
                None => return Ok(()),
            }
        }
    }

    ///
    pub fn slice_iterate_mut<F>(&mut self, mut f: F) -> Result<(), Error>
    where
        F: FnMut(&mut [u8]) -> Result<(), Error>,
    {
        if self.is_empty() {
            return Ok(());
        }

        let mut slice;
        let mut next = self.start_offset();
        loop {
            match self.slice_mut(next) {
                Some((s, n)) => {
                    slice = s;
                    next = n;
                    if slice.is_empty() {
                        continue;
                    }
                    f(slice)?;
                }
                None => return Ok(()),
            }
        }
    }
}
