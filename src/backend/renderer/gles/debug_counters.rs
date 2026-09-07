//! Process-global VRAM lifecycle counters for leak hunting.
//!
//! A GPU-side object whose creation count outpaces its destruction count without
//! a tracked CPU-side handle is invisible to ordinary heap profilers: the GL
//! driver owns the memory, not the process heap. These counters make that drift
//! observable. Each GL object class the renderer allocates (textures, EGLImages,
//! renderbuffers, GL buffer objects, framebuffers) gets a created/destroyed pair,
//! and every deferred-destruction queue variant tracks submissions, explicit
//! drains, and discarded requests. Outstanding requests are
//! `queued - drained - discarded`. Discarding a request does not prove GL deletion;
//! context retirement can release objects without an explicit deletion call.
//!
//! The counters are deliberately process-global rather than per-renderer or
//! per-context: GL resources are freed lazily on whichever shared context runs
//! `cleanup` next, so a renderer can create objects another renderer destroys.
//! Per-instance counters would split that accounting and hide the very drift we
//! are trying to see. All contexts and threads share one set of atomics.
//!
//! Increments use `Relaxed` ordering. We only need monotonic per-counter totals
//! for a coarse snapshot diff, not cross-counter happens-before, so the cheapest
//! ordering is correct here and keeps the instrumentation off the hot path's
//! critical timing.

use std::backtrace::Backtrace;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

macro_rules! counters {
    ($($name:ident),* $(,)?) => {
        $( static $name: AtomicU64 = AtomicU64::new(0); )*
    };
}

counters! {
    TEXTURES_CREATED, TEXTURES_DESTROYED,
    EGL_IMAGES_CREATED, EGL_IMAGES_DESTROYED,
    RENDERBUFFERS_CREATED, RENDERBUFFERS_DESTROYED,
    BUFFERS_CREATED, BUFFERS_DESTROYED,
    FRAMEBUFFERS_CREATED, FRAMEBUFFERS_DESTROYED,

    QUEUED_TEXTURE, QUEUED_FRAMEBUFFER, QUEUED_RENDERBUFFER,
    QUEUED_EGL_IMAGE, QUEUED_MAPPING, QUEUED_PROGRAM, QUEUED_SYNC,

    DRAINED_TEXTURE, DRAINED_FRAMEBUFFER, DRAINED_RENDERBUFFER,
    DRAINED_EGL_IMAGE, DRAINED_MAPPING, DRAINED_PROGRAM, DRAINED_SYNC,

    DISCARDED_TEXTURE,
    DISCARDED_FRAMEBUFFER,
    DISCARDED_RENDERBUFFER,
    DISCARDED_EGL_IMAGE,
    DISCARDED_MAPPING,
    DISCARDED_PROGRAM,
    DISCARDED_SYNC,

    // Diagnostic: `EGLImage`s reclaimed on `import_dmabuf`'s `import_egl_image`
    // error path. `create_image_from_dmabuf` allocates the handle before the bind;
    // when the bind fails the handle has no owning `GlesTexture`. This counts how
    // often that path fired — a non-trivial value confirms it as a leak source.
    EGL_IMAGES_FREED_ON_IMPORT_ERROR,
}

#[inline]
fn inc(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

impl super::debug_queue::Accounting for super::CleanupResource {
    fn queued(&self) {
        use super::CleanupResource::*;
        inc(match self {
            Texture(_) => &QUEUED_TEXTURE,
            FramebufferObject(_) => &QUEUED_FRAMEBUFFER,
            RenderbufferObject(_) => &QUEUED_RENDERBUFFER,
            EGLImage(_) => &QUEUED_EGL_IMAGE,
            Mapping(_, _) => &QUEUED_MAPPING,
            Program(_) => &QUEUED_PROGRAM,
            Sync(_) => &QUEUED_SYNC,
        });
    }

    fn discarded(&self) {
        use super::CleanupResource::*;
        inc(match self {
            Texture(_) => &DISCARDED_TEXTURE,
            FramebufferObject(_) => &DISCARDED_FRAMEBUFFER,
            RenderbufferObject(_) => &DISCARDED_RENDERBUFFER,
            EGLImage(_) => &DISCARDED_EGL_IMAGE,
            Mapping(_, _) => &DISCARDED_MAPPING,
            Program(_) => &DISCARDED_PROGRAM,
            Sync(_) => &DISCARDED_SYNC,
        });
    }
}

pub(super) fn texture_created() {
    inc(&TEXTURES_CREATED);
}
pub(super) fn texture_destroyed() {
    inc(&TEXTURES_DESTROYED);
}
pub(crate) fn egl_image_created(handle: usize) {
    inc(&EGL_IMAGES_CREATED);
    egl_image_register(handle);
}
pub(crate) fn egl_image_destroyed(handle: usize) {
    inc(&EGL_IMAGES_DESTROYED);
    egl_image_deregister(handle);
}
pub(crate) fn egl_image_freed_on_import_error() {
    inc(&EGL_IMAGES_FREED_ON_IMPORT_ERROR);
}

/// Public hook for code that destroys an `EGLImage` outside the renderer's own
/// cleanup queue — e.g. cosmic-comp's dmabuf-import validation, which creates an
/// image with `create_image_from_dmabuf` (counted via `egl_image_created`) and
/// destroys it immediately with a raw `DestroyImageKHR`. Without this the create
/// is counted but the destroy is not, so `egl_images_created` drifts up forever
/// (one per validated client buffer) and masquerades as a leak. Call this right
/// after the raw destroy to keep the counters and allocation-site registry
/// balanced.
pub fn note_egl_image_destroyed(handle: usize) {
    egl_image_destroyed(handle);
}

/// VRAM-leak instrumentation: per-`EGLImage` creation backtraces, keyed by the
/// raw handle. An image created (`egl_image_created`) but never destroyed
/// (`egl_image_destroyed`) leaves a surviving entry whose backtrace is the
/// allocation site of a leaked handle — the EGLImage analogue of the `Dmabuf`
/// clone-site registry. Off unless `COSMIC_DMABUF_TRACE` is set, so default
/// builds pay nothing.
fn egl_image_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("COSMIC_DMABUF_TRACE").is_some())
}

/// Live tracked `EGLImage`s: `handle -> creation backtrace`. One entry per live
/// handle, so the surviving entries after a leak are exactly the orphaned images.
/// The backtrace is an `Arc` so a census can snapshot cheaply and symbolise
/// outside the lock.
fn egl_image_registry() -> &'static Mutex<HashMap<usize, Arc<Backtrace>>> {
    static REG: OnceLock<Mutex<HashMap<usize, Arc<Backtrace>>>> = OnceLock::new();
    REG.get_or_init(|| Mutex::new(HashMap::new()))
}

fn egl_image_register(handle: usize) {
    if !egl_image_trace_enabled() {
        return;
    }
    let bt = Arc::new(Backtrace::force_capture());
    if let Ok(mut reg) = egl_image_registry().lock() {
        reg.insert(handle, bt);
    }
}

fn egl_image_deregister(handle: usize) {
    if !egl_image_trace_enabled() {
        return;
    }
    if let Ok(mut reg) = egl_image_registry().lock() {
        reg.remove(&handle);
    }
}

/// VRAM-leak instrumentation: snapshot of every live tracked `EGLImage` as
/// `(handle, backtrace)`. Empty unless `COSMIC_DMABUF_TRACE` is set. Backtraces
/// are symbolised outside the registry lock so a census never stalls the import
/// path on symbolisation.
pub fn debug_egl_image_sites() -> Vec<(usize, String)> {
    let entries: Vec<(usize, Arc<Backtrace>)> = match egl_image_registry().lock() {
        Ok(reg) => reg.iter().map(|(handle, bt)| (*handle, bt.clone())).collect(),
        Err(_) => return Vec::new(),
    };
    entries
        .into_iter()
        .map(|(handle, bt)| (handle, bt.to_string()))
        .collect()
}
pub(super) fn renderbuffer_created() {
    inc(&RENDERBUFFERS_CREATED);
}
pub(super) fn renderbuffer_destroyed() {
    inc(&RENDERBUFFERS_DESTROYED);
}
pub(super) fn buffer_created() {
    inc(&BUFFERS_CREATED);
}
pub(super) fn buffer_destroyed() {
    inc(&BUFFERS_DESTROYED);
}
pub(super) fn framebuffer_created() {
    inc(&FRAMEBUFFERS_CREATED);
}
pub(super) fn framebuffer_destroyed() {
    inc(&FRAMEBUFFERS_DESTROYED);
}

pub(super) fn drained_texture() {
    inc(&DRAINED_TEXTURE);
}
pub(super) fn drained_framebuffer() {
    inc(&DRAINED_FRAMEBUFFER);
}
pub(super) fn drained_renderbuffer() {
    inc(&DRAINED_RENDERBUFFER);
}
pub(super) fn drained_egl_image() {
    inc(&DRAINED_EGL_IMAGE);
}
pub(super) fn drained_mapping() {
    inc(&DRAINED_MAPPING);
}
pub(super) fn drained_program() {
    inc(&DRAINED_PROGRAM);
}
pub(super) fn drained_sync() {
    inc(&DRAINED_SYNC);
}

/// Snapshot of the process-global GPU-object lifecycle counters.
///
/// All fields are monotonic totals across contexts and threads. Outstanding
/// cleanup requests are `queued - drained - discarded`. Discarded requests include
/// sends to retired queues and pending entries dropped with their receiver.
/// They do not count as explicit GL destruction. Creation/destruction differences
/// therefore require separate interpretation when contexts have retired.
///
/// Counts are reads of independent `Relaxed` atomics, so a snapshot is not a
/// consistent instant across counters — a destroy may be observed before its
/// paired create within one snapshot. That skew is at most a handful of
/// in-flight operations and does not affect the unbounded-growth signal a leak
/// produces over time.
#[derive(Debug, Clone, Copy)]
pub struct VramCounters {
    /// `GlesTextureInternal` GL texture names allocated.
    pub textures_created: u64,
    /// GL texture names freed via the cleanup queue.
    pub textures_destroyed: u64,
    /// `EGLImage` handles created (dmabuf import and wl_drm/EGL buffer import).
    pub egl_images_created: u64,
    /// `EGLImage` handles destroyed via `DestroyImageKHR`.
    pub egl_images_destroyed: u64,
    /// GL renderbuffer names allocated.
    pub renderbuffers_created: u64,
    /// GL renderbuffer names freed.
    pub renderbuffers_destroyed: u64,
    /// GL buffer-object names allocated (read-back PBOs and the vertex buffers).
    pub buffers_created: u64,
    /// GL buffer-object names freed.
    pub buffers_destroyed: u64,
    /// GL framebuffer-object names allocated.
    pub framebuffers_created: u64,
    /// GL framebuffer-object names freed.
    pub framebuffers_destroyed: u64,

    /// Texture cleanup submissions, including sends rejected by a retired receiver.
    pub queued_texture: u64,
    /// `CleanupResource::FramebufferObject` items queued.
    pub queued_framebuffer: u64,
    /// `CleanupResource::RenderbufferObject` items queued.
    pub queued_renderbuffer: u64,
    /// `CleanupResource::EGLImage` items queued.
    pub queued_egl_image: u64,
    /// `CleanupResource::Mapping` items queued.
    pub queued_mapping: u64,
    /// `CleanupResource::Program` items queued.
    pub queued_program: u64,
    /// `CleanupResource::Sync` items queued.
    pub queued_sync: u64,

    /// `CleanupResource::Texture` items drained and freed in `cleanup`.
    pub drained_texture: u64,
    /// `CleanupResource::FramebufferObject` items drained.
    pub drained_framebuffer: u64,
    /// `CleanupResource::RenderbufferObject` items drained.
    pub drained_renderbuffer: u64,
    /// `CleanupResource::EGLImage` items drained.
    pub drained_egl_image: u64,
    /// `CleanupResource::Mapping` items drained.
    pub drained_mapping: u64,
    /// `CleanupResource::Program` items drained.
    pub drained_program: u64,
    /// `CleanupResource::Sync` items drained.
    pub drained_sync: u64,

    /// Texture cleanup requests dropped without explicit GL deletion.
    pub discarded_texture: u64,
    /// Framebuffer cleanup requests dropped without explicit GL deletion.
    pub discarded_framebuffer: u64,
    /// Renderbuffer cleanup requests dropped without explicit GL deletion.
    pub discarded_renderbuffer: u64,
    /// EGLImage cleanup requests dropped without explicit GL deletion.
    pub discarded_egl_image: u64,
    /// Mapping cleanup requests dropped without explicit GL deletion.
    pub discarded_mapping: u64,
    /// Program cleanup requests dropped without explicit GL deletion.
    pub discarded_program: u64,
    /// Sync cleanup requests dropped without explicit GL deletion.
    pub discarded_sync: u64,

    /// `EGLImage`s reclaimed on `import_dmabuf`'s `import_egl_image` error path
    /// (see [`EGL_IMAGES_FREED_ON_IMPORT_ERROR`]).
    pub egl_images_freed_on_import_error: u64,
}

/// Take a snapshot of the process-global GPU-object lifecycle counters.
///
/// See [`VramCounters`] for how to read the snapshot. Cheap and lock-free; safe
/// to call from any thread, including a signal-driven census dump.
pub fn vram_counters() -> VramCounters {
    let load = |c: &AtomicU64| c.load(Ordering::Relaxed);
    VramCounters {
        textures_created: load(&TEXTURES_CREATED),
        textures_destroyed: load(&TEXTURES_DESTROYED),
        egl_images_created: load(&EGL_IMAGES_CREATED),
        egl_images_destroyed: load(&EGL_IMAGES_DESTROYED),
        renderbuffers_created: load(&RENDERBUFFERS_CREATED),
        renderbuffers_destroyed: load(&RENDERBUFFERS_DESTROYED),
        buffers_created: load(&BUFFERS_CREATED),
        buffers_destroyed: load(&BUFFERS_DESTROYED),
        framebuffers_created: load(&FRAMEBUFFERS_CREATED),
        framebuffers_destroyed: load(&FRAMEBUFFERS_DESTROYED),

        queued_texture: load(&QUEUED_TEXTURE),
        queued_framebuffer: load(&QUEUED_FRAMEBUFFER),
        queued_renderbuffer: load(&QUEUED_RENDERBUFFER),
        queued_egl_image: load(&QUEUED_EGL_IMAGE),
        queued_mapping: load(&QUEUED_MAPPING),
        queued_program: load(&QUEUED_PROGRAM),
        queued_sync: load(&QUEUED_SYNC),

        drained_texture: load(&DRAINED_TEXTURE),
        drained_framebuffer: load(&DRAINED_FRAMEBUFFER),
        drained_renderbuffer: load(&DRAINED_RENDERBUFFER),
        drained_egl_image: load(&DRAINED_EGL_IMAGE),
        drained_mapping: load(&DRAINED_MAPPING),
        drained_program: load(&DRAINED_PROGRAM),
        drained_sync: load(&DRAINED_SYNC),

        discarded_texture: load(&DISCARDED_TEXTURE),
        discarded_framebuffer: load(&DISCARDED_FRAMEBUFFER),
        discarded_renderbuffer: load(&DISCARDED_RENDERBUFFER),
        discarded_egl_image: load(&DISCARDED_EGL_IMAGE),
        discarded_mapping: load(&DISCARDED_MAPPING),
        discarded_program: load(&DISCARDED_PROGRAM),
        discarded_sync: load(&DISCARDED_SYNC),

        egl_images_freed_on_import_error: load(&EGL_IMAGES_FREED_ON_IMPORT_ERROR),
    }
}
