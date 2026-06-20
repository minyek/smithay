//! Process-global VRAM lifecycle counters for leak hunting.
//!
//! A GPU-side object whose creation count outpaces its destruction count without
//! a tracked CPU-side handle is invisible to ordinary heap profilers: the GL
//! driver owns the memory, not the process heap. These counters make that drift
//! observable. Each GL object class the renderer allocates (textures, EGLImages,
//! renderbuffers, GL buffer objects, framebuffers) gets a created/destroyed pair,
//! and every deferred-destruction queue variant gets a queued/drained pair. A
//! healthy renderer keeps `created - destroyed` and `queued - drained` bounded;
//! unbounded growth in either localises the leak to a specific object class.
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

use std::sync::atomic::{AtomicU64, Ordering};

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
}

#[inline]
fn inc(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

pub(super) fn texture_created() {
    inc(&TEXTURES_CREATED);
}
pub(super) fn texture_destroyed() {
    inc(&TEXTURES_DESTROYED);
}
pub(crate) fn egl_image_created() {
    inc(&EGL_IMAGES_CREATED);
}
pub(crate) fn egl_image_destroyed() {
    inc(&EGL_IMAGES_DESTROYED);
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

pub(super) fn queued_texture() {
    inc(&QUEUED_TEXTURE);
}
pub(super) fn queued_framebuffer() {
    inc(&QUEUED_FRAMEBUFFER);
}
pub(super) fn queued_renderbuffer() {
    inc(&QUEUED_RENDERBUFFER);
}
pub(super) fn queued_egl_image() {
    inc(&QUEUED_EGL_IMAGE);
}
pub(super) fn queued_mapping() {
    inc(&QUEUED_MAPPING);
}
pub(super) fn queued_program() {
    inc(&QUEUED_PROGRAM);
}
pub(super) fn queued_sync() {
    inc(&QUEUED_SYNC);
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
/// Captured for leak hunting: a class whose `*_created` total keeps pulling ahead
/// of its `*_destroyed` total, or a cleanup-queue variant whose `*_queued` total
/// keeps pulling ahead of its `*_drained` total, is the leaking class. All fields
/// are monotonic running totals across every GL context and thread in the
/// process, so two snapshots taken over a window can be diffed to see which class
/// drifted. The struct is `Copy`, so a snapshot is a stable, allocation-free
/// value the caller can log or stash for a later diff.
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

    /// `CleanupResource::Texture` items pushed onto the deferred-destruction queue.
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
    }
}
