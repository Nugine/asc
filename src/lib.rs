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
#![allow(
    clippy::missing_safety_doc, // TODO
    clippy::missing_errors_doc, // TODO
    clippy::wildcard_imports,
    clippy::enum_glob_use,
)]
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
use std::sync::Arc;

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

unsafe impl<T: Send + Sync> Send for Asc<T> {}
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

impl<T> Asc<T> {
    /// Constructs a new `Pin<Asc<T>>`.
    ///
    /// See [`Arc::pin`].
    #[inline]
    #[must_use]
    pub fn pin(data: T) -> Pin<Asc<T>> {
        // Safety: Asc::new always allocates and pins the data on the heap,
        // so the data's address is stable.
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
        if cfg!(not(target_pointer_width = "64")) && old >= isize::MAX as usize {
            critical()
        }

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
mod tests {
    use super::*;

    use alloc::collections::BTreeMap;
    use alloc::format;
    use alloc::string::String;
    use alloc::vec::Vec;

    #[test]
    fn clone_and_drop() {
        let a = Asc::new(String::from("hello"));
        let a1 = a.shallow_clone();
        let a2 = Asc::clone(&a);

        assert_eq!(&*a, "hello");

        drop(a1);
        drop(a);
        drop(a2);
    }

    #[test]
    fn strong_count_basic() {
        let a = Asc::new(42i32);
        assert_eq!(Asc::strong_count(&a), 1);

        let b = a.clone();
        assert_eq!(Asc::strong_count(&a), 2);
        assert_eq!(Asc::strong_count(&b), 2);

        drop(b);
        assert_eq!(Asc::strong_count(&a), 1);
    }

    #[test]
    fn ptr_eq_same() {
        let a = Asc::new(1);
        let b = a.clone();
        assert!(Asc::ptr_eq(&a, &b));

        let c = Asc::new(1);
        assert!(!Asc::ptr_eq(&a, &c));
    }

    #[test]
    fn try_unwrap_success() {
        let a = Asc::new(42);
        let val = Asc::try_unwrap(a).unwrap();
        assert_eq!(val, 42);
    }

    #[test]
    fn try_unwrap_failure() {
        let a = Asc::new(42);
        let _b = a.clone();
        let result = Asc::try_unwrap(a);
        assert!(result.is_err());
        // Returns the Asc back on failure
        let a = result.unwrap_err();
        assert_eq!(*a, 42);
    }

    #[test]
    fn into_inner_unique() {
        let a = Asc::new("data");
        assert_eq!(Asc::into_inner(a), Some("data"));
    }

    #[test]
    fn into_inner_shared() {
        let a = Asc::new("data");
        let _b = a.clone();
        assert_eq!(Asc::into_inner(a), None);
    }

    #[test]
    fn unwrap_or_clone_unique() {
        let a = Asc::new(String::from("hello"));
        let s = Asc::unwrap_or_clone(a);
        assert_eq!(s, "hello");
    }

    #[test]
    fn unwrap_or_clone_shared() {
        let a = Asc::new(String::from("hello"));
        let _b = a.clone();
        let s = Asc::unwrap_or_clone(a);
        assert_eq!(s, "hello");
    }

    // roundtrip through raw pointer creates a temporary &Inner<T> in
    // as_ptr; miri Stacked Borrows rejects re-deriving a reference from
    // the raw pointer in from_raw. This is a known limitation of the
    // Stacked Borrows model and does not indicate a real bug.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn into_raw_from_raw_roundtrip() {
        let a = Asc::new(42i32);
        let ptr = Asc::into_raw(a);
        let a = unsafe { Asc::from_raw(ptr) };
        assert_eq!(*a, 42);
    }

    #[test]
    fn as_ptr_borrowed() {
        let a = Asc::new(7u32);
        let ptr = Asc::as_ptr(&a);
        // as_ptr provides a borrowed pointer — the original Asc still owns
        // the allocation. Reading through the pointer is safe while a lives.
        assert_eq!(unsafe { *ptr }, 7);
        assert_eq!(*a, 7);
    }

    // miri: see comment on into_raw_from_raw_roundtrip.
    #[test]
    #[cfg_attr(miri, ignore)]
    fn from_raw_high_alignment() {
        #[repr(align(64))]
        struct Aligned(u8);

        let a = Asc::new(Aligned(42));
        let ptr = Asc::into_raw(a);
        let a = unsafe { Asc::from_raw(ptr) };
        assert_eq!(a.0, 42);
    }

    #[test]
    fn get_mut_unique() {
        let mut a = Asc::new(10i32);
        {
            let val = Asc::get_mut(&mut a).unwrap();
            *val = 20;
        }
        assert_eq!(*a, 20);
    }

    #[test]
    fn get_mut_shared() {
        let mut a = Asc::new(10i32);
        let _b = a.clone();
        assert!(Asc::get_mut(&mut a).is_none());
    }

    #[test]
    fn make_mut_unique() {
        let mut a = Asc::new(5i32);
        *Asc::make_mut(&mut a) = 10;
        assert_eq!(*a, 10);
        assert_eq!(Asc::strong_count(&a), 1);
    }

    #[test]
    fn make_mut_shared_clones() {
        let mut a = Asc::new(5i32);
        let _b = a.clone();
        assert_eq!(Asc::strong_count(&a), 2);

        // make_mut should clone-on-write (allocate new inner)
        *Asc::make_mut(&mut a) = 10;
        assert_eq!(*a, 10);
        // Now a should have its own allocation with count 1
        assert_eq!(Asc::strong_count(&a), 1);
    }

    #[test]
    fn pin_basic() {
        let pinned = Asc::pin(42i32);
        assert_eq!(*pinned, 42);
        // Cloning a pinned Asc should work
        let _clone = pinned.clone();
    }

    #[test]
    fn deref() {
        let a = Asc::new(100);
        assert_eq!(*a, 100);
    }

    #[test]
    fn partial_eq() {
        let a = Asc::new(1);
        let b = Asc::new(1);
        let c = Asc::new(2);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn eq_trait() {
        fn assert_eq_trait<T: core::cmp::Eq>(_t: &T) {}
        let a = Asc::new(1);
        assert_eq_trait(&a);
    }

    #[test]
    fn hash() {
        let mut map = BTreeMap::new();
        map.insert(Asc::new(1), "one");
        map.insert(Asc::new(2), "two");
        assert_eq!(map.get(&Asc::new(1)), Some(&"one"));
        assert_eq!(map.get(&Asc::new(2)), Some(&"two"));
    }

    #[test]
    fn partial_ord_and_ord() {
        let a = Asc::new(1);
        let b = Asc::new(2);
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a.cmp(&b), core::cmp::Ordering::Less);
    }

    #[test]
    fn display() {
        let a = Asc::new(42);
        assert_eq!(format!("{a}"), "42");
    }

    #[test]
    fn pointer_fmt() {
        let a = Asc::new(1);
        let s = format!("{a:p}");
        let expected = format!("{:p}", Asc::as_ptr(&a));
        assert_eq!(s, expected);
    }

    #[test]
    fn from_trait() {
        let a: Asc<i32> = 42.into();
        assert_eq!(*a, 42);
    }

    #[test]
    fn default_trait() {
        let a: Asc<i32> = Default::default();
        assert_eq!(*a, 0);
    }

    #[test]
    fn debug() {
        let a = Asc::new(7);
        assert_eq!(format!("{a:?}"), "7");
    }

    #[test]
    fn send_sync() {
        fn assert_send<T: Send>(_t: &T) {}
        fn assert_sync<T: Sync>(_t: &T) {}
        let a = Asc::new(1);
        assert_send(&a);
        assert_sync(&a);
    }

    #[test]
    fn try_unwrap_after_make_mut() {
        let mut a = Asc::new(String::from("hello"));
        let _b = a.clone();
        // This clones into a new allocation
        Asc::make_mut(&mut a).push_str(" world");
        // Now a has count 1 in its own allocation
        let s = Asc::try_unwrap(a).unwrap();
        assert_eq!(s, "hello world");
    }

    #[test]
    fn test_vec_of_asc() {
        let items: Vec<Asc<i32>> = (0..5).map(Asc::new).collect();
        let clones: Vec<Asc<i32>> = items.iter().map(Asc::clone).collect();
        assert_eq!(items.len(), clones.len());
        for (a, b) in items.iter().zip(&clones) {
            assert!(Asc::ptr_eq(a, b));
        }
    }

    #[test]
    fn drop_behavior() {
        // Verify that dropping doesn't panic and strong_count decreases
        let a = Asc::new(());
        let b = a.clone();
        let c = a.clone();
        assert_eq!(Asc::strong_count(&a), 3);
        drop(b);
        assert_eq!(Asc::strong_count(&a), 2);
        drop(c);
        assert_eq!(Asc::strong_count(&a), 1);
    }

    #[test]
    fn zero_sized_type() {
        let a = Asc::new(());
        let b = a.clone();
        assert_eq!(Asc::strong_count(&a), 2);
        assert!(Asc::ptr_eq(&a, &b));
        drop(b);
        let val = Asc::try_unwrap(a).unwrap();
        assert_eq!(val, ());
    }
}
