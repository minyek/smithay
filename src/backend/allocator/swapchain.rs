use std::{
    collections::BTreeMap,
    fmt,
    ops::Deref,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
    },
};

use tracing::instrument;

use crate::backend::allocator::{Allocator, Buffer, Fourcc, Modifier};
use crate::utils::user_data::UserDataMap;

use super::dmabuf::{AsDmabuf, Dmabuf};

pub const SLOT_CAP: usize = 4;

// --- VRAM-leak diagnostic: live-slot registry --------------------------------
// Every `InternalSlot` that gets a buffer allocated registers here, keyed by a
// process-unique id, and deregisters on `Drop`. A slot whose owning `Swapchain`
// was replaced (set_format / resize) yet stays registered is held by an escaped
// `Slot` clone — the residual full-screen render-target leak. The recorded
// acquire backtrace names the render path that allocated it.
static SLOT_UID: AtomicUsize = AtomicUsize::new(0);

struct LiveSlot {
    width: i32,
    height: i32,
    dmabuf_id: Option<u64>,
    acquired_at: String,
}

static LIVE_SLOTS: Mutex<BTreeMap<usize, LiveSlot>> = Mutex::new(BTreeMap::new());

/// Per-size summary of every live (never-dropped) swapchain slot that has had a
/// buffer allocated, for the VRAM-leak census. A size group whose count exceeds
/// the live swapchains' `SLOT_CAP` × output count names stranded render targets.
pub fn debug_live_slots() -> String {
    let map = match LIVE_SLOTS.lock() {
        Ok(m) => m,
        Err(_) => return "live_slots: <poisoned>".into(),
    };
    let mut by_size: BTreeMap<(i32, i32), Vec<String>> = BTreeMap::new();
    for slot in map.values() {
        let id = slot
            .dmabuf_id
            .map(|v| v.to_string())
            .unwrap_or_else(|| "?".into());
        by_size
            .entry((slot.width, slot.height))
            .or_default()
            .push(id);
    }
    let parts = by_size
        .into_iter()
        .map(|((w, h), mut ids)| {
            ids.sort();
            format!("{w}x{h}×{} [{}]", ids.len(), ids.join(","))
        })
        .collect::<Vec<_>>();
    format!("live_slots total={} {{{}}}", map.len(), parts.join("; "))
}

/// Live slots grouped by their acquire-path signature, for drilling into *which*
/// render path stranded them once the census flags a leak.
pub fn debug_live_slot_sites() -> String {
    let map = match LIVE_SLOTS.lock() {
        Ok(m) => m,
        Err(_) => return "live_slot_sites: <poisoned>".into(),
    };
    let mut by_site: BTreeMap<String, usize> = BTreeMap::new();
    for slot in map.values() {
        *by_site.entry(slot.acquired_at.clone()).or_default() += 1;
    }
    let mut parts = by_site.into_iter().collect::<Vec<_>>();
    parts.sort_by(|a, b| b.1.cmp(&a.1));
    parts
        .into_iter()
        .map(|(site, n)| format!("{n}× {site}"))
        .collect::<Vec<_>>()
        .join(" | ")
}

/// Reduce a full backtrace to the cosmic-comp / smithay-drm frames that identify
/// the acquire path, as a single `a <- b <- c` line.
fn acquire_signature() -> String {
    let text = std::backtrace::Backtrace::force_capture().to_string();
    let mut frames: Vec<String> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let relevant = line.contains("cosmic_comp::backend::kms")
            || line.contains("drm::output")
            || line.contains("drm::compositor")
            || line.contains("allow_frame_flags")
            || line.contains("apply_config")
            || line.contains("initialize_output")
            || line.contains("use_mode")
            || line.contains("try_to_restore")
            || line.contains("submit_composited_frame");
        if !relevant {
            continue;
        }
        let sym = line.split_once(": ").map(|(_, s)| s).unwrap_or(line);
        let sym = sym.split("::h").next().unwrap_or(sym);
        let short = sym
            .rsplit("::")
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("::");
        if !short.is_empty() && !frames.contains(&short) {
            frames.push(short);
        }
        if frames.len() >= 5 {
            break;
        }
    }
    if frames.is_empty() {
        "<other>".into()
    } else {
        frames.join(" <- ")
    }
}

/// Swapchain handling a fixed set of re-usable buffers e.g. for scan-out.
///
/// ## How am I supposed to use this?
///
/// To do proper buffer management, most compositors do so called double-buffering.
/// Which means you use two buffers, one that is currently presented (the front buffer)
/// and one that is currently rendered to (the back buffer). After each rendering operation
/// you swap the buffers around, the old front buffer becomes the new back buffer, while
/// the new front buffer is displayed to the user. This avoids showing the user rendering
/// artifacts doing rendering.
///
/// There are also reasons to do triple-buffering, e.g. if you swap operation takes a
/// unspecified amount of time. In that case you have one buffer, that is currently
/// displayed, one that is done drawing and about to be swapped in and another one,
/// which you can use to render currently.
///
/// Re-using and managing these buffers becomes increasingly complex the more buffers you
/// introduce, which is where `Swapchain` comes into play.
///
/// `Swapchain` allocates buffers for you and transparently re-created them, e.g. when resizing.
/// All you tell the swapchain is: *"Give me the next free buffer"* (by calling [`acquire`](Swapchain::acquire)).
/// You then hold on to the returned buffer during rendering and swapping and free it once it is displayed.
/// Efficient re-use of the buffers is done by the swapchain.
///
/// If you have associated resources for each buffer that can be reused (e.g. framebuffer `Handle`s for a `DrmDevice`),
/// you can store then in the `Slot`s userdata field. If a buffer is re-used, its userdata is preserved for the next time
/// it is returned by `acquire()`.
pub struct Swapchain<A: Allocator> {
    /// Allocator used by the swapchain
    pub allocator: A,

    width: u32,
    height: u32,
    fourcc: Fourcc,
    modifiers: Vec<Modifier>,

    slots: [Arc<InternalSlot<A::Buffer>>; SLOT_CAP],
}

impl<A: Allocator> fmt::Debug for Swapchain<A> {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt.debug_struct("Swapchain")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("fourcc", &self.fourcc)
            .field("modifiers", &self.modifiers)
            .finish_non_exhaustive()
    }
}

/// Slot of a swapchain containing an allocated buffer and its userdata.
///
/// The buffer is marked for re-use once all copies are dropped.
/// Holding on to this struct will block the buffer in the swapchain.
#[derive(Debug)]
pub struct Slot<B: Buffer>(Arc<InternalSlot<B>>);

#[derive(Debug)]
struct InternalSlot<B: Buffer> {
    buffer: Option<B>,
    acquired: AtomicBool,
    age: AtomicU8,
    userdata: UserDataMap,
    uid: usize,
}

impl<B: Buffer> Drop for InternalSlot<B> {
    fn drop(&mut self) {
        if let Ok(mut map) = LIVE_SLOTS.lock() {
            map.remove(&self.uid);
        }
    }
}

impl<B: Buffer> Slot<B> {
    /// Retrieve userdata for this slot.
    pub fn userdata(&self) -> &UserDataMap {
        &self.0.userdata
    }

    /// Retrieve the age of the buffer
    pub fn age(&self) -> u8 {
        self.0.age.load(Ordering::SeqCst)
    }
}

impl<B: Buffer> Default for InternalSlot<B> {
    fn default() -> Self {
        InternalSlot {
            buffer: None,
            acquired: AtomicBool::new(false),
            age: AtomicU8::new(0),
            userdata: UserDataMap::new(),
            uid: SLOT_UID.fetch_add(1, Ordering::Relaxed),
        }
    }
}

impl<B: Buffer> Deref for Slot<B> {
    type Target = B;
    fn deref(&self) -> &B {
        Option::as_ref(&self.0.buffer).unwrap()
    }
}

impl<B: Buffer + AsDmabuf> AsDmabuf for Slot<B> {
    type Error = <B as AsDmabuf>::Error;

    fn export(&self) -> Result<super::dmabuf::Dmabuf, Self::Error> {
        let maybe_dmabuf = self.userdata().get::<Dmabuf>();
        if maybe_dmabuf.is_none() {
            let dmabuf = (**self).export()?;
            self.userdata().insert_if_missing_threadsafe(|| dmabuf);
        }

        let dmabuf = self.userdata().get::<Dmabuf>().cloned().unwrap();
        // Link this slot's diagnostic registry entry to its exported dmabuf id so
        // the census can cross-reference orphans against the main renderer cache.
        if let Ok(mut map) = LIVE_SLOTS.lock() {
            if let Some(entry) = map.get_mut(&self.0.uid) {
                entry.dmabuf_id = Some(dmabuf.debug_id());
            }
        }
        Ok(dmabuf)
    }
}

impl<B: Buffer> Drop for Slot<B> {
    fn drop(&mut self) {
        self.0.acquired.store(false, Ordering::SeqCst);
    }
}

impl<A> Swapchain<A>
where
    A: Allocator,
{
    /// Create a new swapchain with the desired allocator, dimensions and pixel format for the created buffers.
    pub fn new(
        allocator: A,
        width: u32,
        height: u32,
        fourcc: Fourcc,
        modifiers: Vec<Modifier>,
    ) -> Swapchain<A> {
        Swapchain {
            allocator,
            width,
            height,
            fourcc,
            modifiers,
            slots: Default::default(),
        }
    }

    /// Acquire a new slot from the swapchain, if one is still free.
    ///
    /// The swapchain has an internal maximum of four re-usable buffers.
    /// This function returns the first free one.
    #[instrument(level = "trace", skip_all, err)]
    #[profiling::function]
    pub fn acquire(&mut self) -> Result<Option<Slot<A::Buffer>>, A::Error> {
        if let Some(free_slot) = self
            .slots
            .iter_mut()
            .find(|s| !s.acquired.swap(true, Ordering::SeqCst))
        {
            if free_slot.buffer.is_none() {
                let free_slot = Arc::get_mut(free_slot).expect("Acquired was false, but Arc is not unique?");
                match self
                    .allocator
                    .create_buffer(self.width, self.height, self.fourcc, &self.modifiers)
                {
                    Ok(buffer) => {
                        let size = buffer.size();
                        if let Ok(mut map) = LIVE_SLOTS.lock() {
                            map.insert(
                                free_slot.uid,
                                LiveSlot {
                                    width: size.w,
                                    height: size.h,
                                    dmabuf_id: None,
                                    acquired_at: acquire_signature(),
                                },
                            );
                        }
                        free_slot.buffer = Some(buffer);
                    }
                    Err(err) => {
                        free_slot.acquired.store(false, Ordering::SeqCst);
                        return Err(err);
                    }
                }
            }
            assert!(free_slot.buffer.is_some());
            return Ok(Some(Slot(free_slot.clone())));
        }

        // no free slots
        Ok(None)
    }

    /// Mark a given buffer as submitted.
    ///
    /// This might effect internal data (e.g. buffer age) and may only be called,
    /// the buffer may not be used for rendering anymore.
    /// You may hold on to it, if you require keeping it alive.
    ///
    /// Buffers can always just be safely discarded by dropping them, but not
    /// calling this function before may affect performance characteristics
    /// (e.g. by not tracking the buffer age).
    pub fn submitted(&mut self, slot: &Slot<A::Buffer>) {
        // don't mess up the state, if the user submitted and old buffer, after e.g. a resize
        if !self.slots.iter().any(|other| Arc::ptr_eq(&slot.0, other)) {
            return;
        }

        slot.0.age.store(1, Ordering::SeqCst);
        for other_slot in &mut self.slots {
            if !Arc::ptr_eq(other_slot, &slot.0) && other_slot.buffer.is_some() {
                let res = other_slot
                    .age
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |age| {
                        if age > 0 { age.checked_add(1) } else { Some(0) }
                    });
                // If the age overflows the slot was not used for a long time. Lets clear it
                if res.is_err() {
                    *other_slot = Default::default();
                }
            }
        }
    }

    /// Change the dimensions of newly returned buffers.
    ///
    /// Already obtained buffers are unaffected and will be cleaned up on drop.
    pub fn resize(&mut self, width: u32, height: u32) {
        if self.width == width && self.height == height {
            return;
        }

        self.width = width;
        self.height = height;
        self.slots = Default::default();
    }

    /// Remove all internally cached buffers.
    pub fn reset_buffers(&mut self) {
        for slot in &mut self.slots {
            *slot = Default::default();
        }
    }

    /// Reset the age for each buffer.
    ///
    /// Resetting the buffer age will discard all damage information and force a
    /// full redraw for the next frame.
    pub fn reset_buffer_ages(&mut self) {
        for slot in &mut self.slots {
            match Arc::get_mut(slot) {
                Some(slot) => slot.age = AtomicU8::new(0),
                None => *slot = Default::default(),
            }
        }
    }

    /// VRAM-leak instrumentation: the debug-id of each slot's exported dmabuf, or
    /// `None` for slots that have not been exported. Lets the census map a cached
    /// render-target import back to a live swapchain slot.
    pub fn debug_slot_dmabuf_ids(&self) -> Vec<Option<u64>> {
        self.slots
            .iter()
            .map(|slot| slot.userdata.get::<Dmabuf>().map(|d| d.debug_id()))
            .collect()
    }

    /// Get set format
    pub fn format(&self) -> Fourcc {
        self.fourcc
    }

    /// Get allowed modifiers
    pub fn modifiers(&self) -> &[Modifier] {
        &self.modifiers
    }
}
