//! A template for creating Rust open-source repo on GitHub
//!
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(docsrs, allow(unused_attributes))]
#![deny(missing_docs)]

use fmmap::{MmapFileMut, MmapFileMutExt, MmapFileExt};
use std::{path::{Path, PathBuf}, sync::atomic::{AtomicU64, Ordering}};

const DEFAULT_CAPACITY: usize = 64;

const DEFAULT_TAG: &str = "buffer";

/// Config auto mmap
pub struct AutoMmapMeta
{
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
pub struct Buffer
{
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
    /// 
    pub fn new(capacity: usize, mut tag: &'static str) -> Self {
        let capacity = capacity.max(DEFAULT_CAPACITY);
        if tag.is_empty() {
            tag = DEFAULT_TAG;
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
            dir: path.map(|p| p.as_ref().to_path_buf()).unwrap_or_else(std::env::temp_dir),
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


    /// Grow would grow the buffer to have at least n more bytes. In case the buffer is at capacity, it
    /// would reallocate twice the size of current capacity + n, to ensure n bytes can be written to the
    /// buffer without further allocation. In UseMmap mode, this might result in underlying file
    /// expansion.
    pub fn grow(&mut self, n: usize) -> Result<(), ()> {
        let size = (self.offset.load(Ordering::Relaxed) as usize) + n;
        if let Some(max_size) = self.max_size {
            if max_size > 0 && size > max_size {
                return Err(());
            }
        }

        if size < self.cur_size {
            return Ok(());
        }


        self.cur_size += (self.cur_size + n)  // Calculate new capacity.
            .min(1 << 30) // Don't allocate more than 1GB at a time.
            .max(n); // Allocate at least n, even if it exceeds the 1GB limit above.

        match self.buf_type {
            BufferType::Memory => {
                // If autoMmap gets triggered, copy the slice over to an mmaped file.
                if let Some(ref meta) = self.auto_mmap_meta {
                    if meta.after > 0 && self.cur_size > meta.after {
                        self.buf_type = BufferType::Mmap;
                        // TODO: add more useful methods in fmmap crate
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
                self.mmap_file.truncate(self.cur_size as u64)
            },
            BufferType::Mmap => {
                // Truncate and remap the underlying file.
                self.mmap_file.truncate(self.cur_size as u64)
            },
        }
    }
}

