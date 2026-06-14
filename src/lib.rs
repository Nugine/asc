//! Atomic Strong Count.
//!
//! [`Asc`] is a drop-in replacement for [`std::sync::Arc`] when you don't
//! need weak references. It provides shared, thread-safe ownership of
//! heap-allocated data.
//!
//! # Key differences from [`Arc`]
//!
//! * No [`Weak`] references — the allocation is freed as soon as the last
//!   [`Asc`] is dropped.
//! * No allocator parameter — always uses the global allocator.
//!   Custom allocators depend on the unstable [`allocator_api`] feature
//!   and will be considered once it stabilizes.
//! * [`Asc::from_raw`], [`Asc::as_ptr`], [`Asc::into_raw`], and
//!   [`Asc::get_mut_unchecked`] are `const` functions.
//!
//! # Optional features
//!
//! * **`serde`** — Enables [`serde`] serialization and deserialization.
//! * **`std`** — Enables [`UnwindSafe`] and [`RefUnwindSafe`]
//!   implementations for [`Asc`].
//! * **`unstable`** — Enables [`CoerceUnsized`] and [`DispatchFromDyn`]
//!   implementations, allowing [`Asc<T>`] to be coerced to
//!   [`Asc<dyn Trait>`]. Requires a nightly compiler.
//!
//! # Example
//!
//! ```
//! use asc::Asc;
//!
//! let a: Asc<i32> = Asc::new(42);
//! let b = a.clone();
//!
//! assert_eq!(Asc::strong_count(&a), 2);
//! assert_eq!(*b, 42);
//!
//! // try_unwrap fails while other references exist
//! assert!(Asc::try_unwrap(a.clone()).is_err());
//!
//! drop(b);
//!
//! // try_unwrap succeeds with the last reference
//! let value = Asc::try_unwrap(a).unwrap();
//! assert_eq!(value, 42);
//! ```
//!
//! [`Weak`]: std::sync::Weak
//! [`UnwindSafe`]: std::panic::UnwindSafe
//! [`RefUnwindSafe`]: std::panic::RefUnwindSafe
//! [`CoerceUnsized`]: core::ops::CoerceUnsized
//! [`DispatchFromDyn`]: core::ops::DispatchFromDyn
//! [`allocator_api`]: https://github.com/rust-lang/rust/issues/32838
#![deny(
    clippy::all,
    clippy::cargo, //
    clippy::pedantic, //
    clippy::as_conversions,
    clippy::float_arithmetic,
    clippy::arithmetic_side_effects,
    clippy::must_use_candidate,
    clippy::missing_inline_in_public_items,
    clippy::missing_const_for_fn
)]
#![allow(clippy::wildcard_imports, clippy::enum_glob_use)]
//
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(
    feature = "unstable",
    feature(unsize, dispatch_from_dyn, coerce_unsized)
)]
#![no_std]

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

use core::alloc::Layout;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::mem;
use core::mem::ManuallyDrop;
use core::ops::Deref;
use core::pin::Pin;
use core::ptr;
use core::ptr::NonNull;
use core::sync::atomic::fence;
use core::sync::atomic::AtomicUsize;
use core::sync::atomic::Ordering::*;

use alloc::boxed::Box;

// Bring Arc into scope for rustdoc intra-doc links.
#[cfg(doc)]
use alloc::sync::Arc;

#[cfg(feature = "unstable")]
use core::marker::Unsize;
#[cfg(feature = "unstable")]
use core::ops::DispatchFromDyn;

#[cfg(feature = "std")]
use std::panic::{RefUnwindSafe, UnwindSafe};

/// Atomic Strong Count.
///
/// [`Asc`] is a drop-in replacement for
/// [`Arc`](https://doc.rust-lang.org/nightly/std/sync/struct.Arc.html)
/// when you don't need weak references.
pub struct Asc<T: ?Sized> {
    inner: NonNull<Inner<T>>,
    _marker: PhantomData<T>,
}

// Safety: Asc<T> provides the same shared-ownership semantics as Arc<T>.
// When T: Send + Sync, sharing T through Asc across thread boundaries is
// sound because all accesses go through the same atomic reference-counting
// and borrow-checking discipline as std's Arc.
unsafe impl<T: Send + Sync> Send for Asc<T> {}

// Safety: Asc<T> provides the same shared-ownership semantics as Arc<T>.
// &Asc<T> gives access to &T via Deref, so if &T: Sync then &Asc<T>: Sync.
unsafe impl<T: Send + Sync> Sync for Asc<T> {}

#[cfg(feature = "std")]
impl<T: RefUnwindSafe + ?Sized> UnwindSafe for Asc<T> {}

#[cfg(feature = "std")]
impl<T: RefUnwindSafe + ?Sized> RefUnwindSafe for Asc<T> {}

#[cfg(feature = "unstable")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> core::ops::CoerceUnsized<Asc<U>> for Asc<T> {}

#[cfg(feature = "unstable")]
impl<T: ?Sized + Unsize<U>, U: ?Sized> DispatchFromDyn<Asc<U>> for Asc<T> {}

#[repr(C)]
struct Inner<T: ?Sized> {
    strong: AtomicUsize,
    data: T,
}

unsafe fn box_from_nonnull<T: ?Sized>(p: NonNull<T>) -> Box<T> {
    Box::from_raw(p.as_ptr())
}

fn box_into_nonnull<T>(b: Box<T>) -> NonNull<T> {
    unsafe { NonNull::new_unchecked(Box::into_raw(b)) }
}

#[cfg(not(target_pointer_width = "64"))]
#[allow(dead_code)]
#[cold]
fn critical() -> ! {
    struct Bomb {}

    impl Drop for Bomb {
        fn drop(&mut self) {
            panic!("bomb")
        }
    }

    let _bomb = Bomb {};
    panic!("critical failure")
}

#[cfg(not(target_pointer_width = "64"))]
#[inline(always)]
fn check_overflow(old: usize) {
    if old >= isize::MAX as usize {
        critical()
    }
}

#[cfg(target_pointer_width = "64")]
#[inline(always)]
const fn check_overflow(_old: usize) {}

impl<T> Asc<T> {
    /// Constructs a new `Pin<Asc<T>>`.
    ///
    /// See [`Arc::pin`].
    #[inline]
    #[must_use]
    pub fn pin(data: T) -> Pin<Asc<T>> {
        // Safety: Asc::new allocates the data on the heap, so its address
        // is stable for the lifetime of the allocation.
        unsafe { Pin::new_unchecked(Asc::new(data)) }
    }

    /// Constructs a new `Asc<T>`.
    ///
    /// See [`Arc::new`].
    #[inline]
    #[must_use]
    pub fn new(data: T) -> Self {
        let inner = Box::new(Inner {
            strong: AtomicUsize::new(1),
            data,
        });
        Self {
            inner: box_into_nonnull(inner),
            _marker: PhantomData,
        }
    }

    /// Returns the inner value if the `Asc` has exactly one strong reference.
    ///
    /// See [`Arc::try_unwrap`].
    ///
    /// # Errors
    ///
    /// Returns `Err(self)` if there are other strong references to the
    /// same allocation.
    #[inline]
    pub fn try_unwrap(this: Self) -> Result<T, Self> {
        let s = this.strong();
        if s.compare_exchange(1, 0, Relaxed, Relaxed).is_err() {
            return Err(this);
        }
        fence(Acquire);
        unsafe {
            let this = ManuallyDrop::new(this);
            let data = ptr::read(&raw const this.inner.as_ref().data);
            // ManuallyDrop prevents Asc::drop from deallocating Inner<T>.
            // Deallocate the raw memory directly — data was already moved
            // out via ptr::read, so we must not drop T again.
            let layout = Layout::new::<Inner<T>>();
            alloc::alloc::dealloc(this.inner.as_ptr().cast::<u8>(), layout);
            Ok(data)
        }
    }

    /// Returns the inner value if the `Asc` has exactly one strong reference.
    ///
    /// See [`Arc::into_inner`].
    #[inline]
    #[must_use]
    pub fn into_inner(this: Self) -> Option<T> {
        Self::try_unwrap(this).ok()
    }

    /// Reconstructs an `Asc<T>` from a raw pointer previously returned by
    /// [`Asc::into_raw`].
    ///
    /// # Safety
    ///
    /// The pointer must have been returned by a previous call to
    /// [`Asc::into_raw`], and it must not have been passed to
    /// [`Asc::from_raw`] more than once.
    ///
    /// See [`Arc::from_raw`].
    #[inline]
    #[must_use]
    #[allow(clippy::as_conversions)]
    pub const unsafe fn from_raw(ptr: *const T) -> Self {
        let offset = mem::offset_of!(Inner<T>, data);
        let inner = ptr.cast::<u8>().sub(offset) as *mut Inner<T>;
        Self {
            inner: NonNull::new_unchecked(inner),
            _marker: PhantomData,
        }
    }
}

impl<T: ?Sized> Asc<T> {
    const fn strong(&self) -> &AtomicUsize {
        unsafe { &self.inner.as_ref().strong }
    }

    #[inline]
    #[must_use]
    #[allow(clippy::as_conversions)]
    fn shallow_clone(&self) -> Self {
        let s = self.strong();
        let old = s.fetch_add(1, Relaxed);
        check_overflow(old);

        Self {
            inner: self.inner,
            _marker: PhantomData,
        }
    }

    #[inline(never)]
    unsafe fn destroy(&mut self) {
        drop(box_from_nonnull(self.inner));
    }

    /// Returns the number of strong references to this allocation.
    ///
    /// See [`Arc::strong_count`].
    #[inline]
    #[must_use]
    pub fn strong_count(this: &Self) -> usize {
        this.strong().load(Relaxed)
    }

    /// Returns `true` if the two `Asc`s point to the same allocation.
    ///
    /// See [`Arc::ptr_eq`].
    #[inline]
    #[must_use]
    pub fn ptr_eq(this: &Self, other: &Self) -> bool {
        ptr::eq(this.inner.as_ptr(), other.inner.as_ptr())
    }

    /// Returns a mutable reference to the inner value if no other `Asc`s
    /// point to the same allocation.
    ///
    /// See [`Arc::get_mut`].
    #[inline]
    #[must_use]
    pub fn get_mut(this: &mut Self) -> Option<&mut T> {
        if this.strong().load(Acquire) == 1 {
            // Safety: strong_count == 1 means this is the only Asc
            // pointing to this allocation, so we have exclusive
            // access to the data. Acquire ordering ensures we see
            // all writes from threads that released their references.
            unsafe { Some(Self::get_mut_unchecked(this)) }
        } else {
            None
        }
    }

    /// Returns a mutable reference to the inner value without checking
    /// the reference count.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no other `Asc` pointers to the same
    /// allocation exist.
    ///
    /// See [`Arc::get_mut_unchecked`].
    #[inline]
    #[must_use]
    pub const unsafe fn get_mut_unchecked(this: &mut Self) -> &mut T {
        &mut this.inner.as_mut().data
    }

    /// Returns a raw pointer to the inner value.
    ///
    /// See [`Arc::as_ptr`].
    #[inline]
    #[must_use]
    pub const fn as_ptr(this: &Self) -> *const T {
        unsafe { ptr::addr_of!(this.inner.as_ref().data) }
    }

    /// Consumes the `Asc` and returns the wrapped raw pointer.
    ///
    /// See [`Arc::into_raw`].
    #[inline]
    #[must_use]
    pub const fn into_raw(this: Self) -> *const T {
        let ptr = Self::as_ptr(&this);
        mem::forget(this);
        ptr
    }
}

impl<T: ?Sized> Deref for Asc<T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &self.inner.as_ref().data }
    }
}

impl<T: ?Sized> Clone for Asc<T> {
    #[inline]
    fn clone(&self) -> Self {
        self.shallow_clone()
    }
}

impl<T: ?Sized> Drop for Asc<T> {
    #[inline]
    fn drop(&mut self) {
        let s = self.strong();
        if s.fetch_sub(1, Release) != 1 {
            return;
        }

        fence(Acquire);
        unsafe { self.destroy() };
    }
}

impl<T: Clone> Asc<T> {
    /// Returns the inner value, cloning it if there are other strong
    /// references.
    ///
    /// See [`Arc::unwrap_or_clone`].
    #[inline]
    #[must_use]
    pub fn unwrap_or_clone(this: Self) -> T {
        Self::try_unwrap(this).unwrap_or_else(|a| T::clone(&a))
    }

    /// Makes a mutable reference to the inner value, cloning the data if
    /// there are other strong references.
    ///
    /// See [`Arc::make_mut`].
    #[inline]
    #[must_use]
    pub fn make_mut(this: &mut Self) -> &mut T {
        let s = this.strong();
        let count = s.load(Acquire);
        if count > 1 {
            *this = Asc::new(T::clone(&**this));
        }
        unsafe { &mut this.inner.as_mut().data }
    }
}

impl<T: fmt::Debug + ?Sized> fmt::Debug for Asc<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <T as fmt::Debug>::fmt(&**self, f)
    }
}

impl<T: ?Sized + fmt::Display> fmt::Display for Asc<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <T as fmt::Display>::fmt(&**self, f)
    }
}

impl<T: ?Sized> fmt::Pointer for Asc<T> {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Pointer::fmt(&Asc::as_ptr(self), f)
    }
}

impl<T: ?Sized + PartialEq> PartialEq for Asc<T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: ?Sized + Eq> Eq for Asc<T> {}

impl<T: ?Sized + Hash> Hash for Asc<T> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T: ?Sized + PartialOrd> PartialOrd for Asc<T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (**self).partial_cmp(&**other)
    }
}

impl<T: ?Sized + Ord> Ord for Asc<T> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        (**self).cmp(&**other)
    }
}

impl<T> From<T> for Asc<T> {
    #[inline]
    fn from(value: T) -> Self {
        Asc::new(value)
    }
}

impl<T: Default> Default for Asc<T> {
    #[inline]
    fn default() -> Self {
        Asc::new(T::default())
    }
}

#[cfg(feature = "serde")]
mod serde_impl {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    impl<'de, T: Deserialize<'de>> Deserialize<'de> for Asc<T> {
        #[inline]
        fn deserialize<D>(deserializer: D) -> Result<Asc<T>, D::Error>
        where
            D: ::serde::de::Deserializer<'de>,
        {
            T::deserialize(deserializer).map(Asc::new)
        }
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
    impl<T: Serialize> Serialize for Asc<T> {
        #[inline]
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: ::serde::ser::Serializer,
        {
            T::serialize(&**self, serializer)
        }
    }
}
#[cfg(test)]
mod tests;
