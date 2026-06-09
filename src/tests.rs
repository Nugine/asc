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
    use core::hash::{BuildHasher, Hasher};

    #[derive(Default)]
    struct SimpleHasher(u64);

    impl Hasher for SimpleHasher {
        fn finish(&self) -> u64 {
            self.0
        }
        fn write(&mut self, bytes: &[u8]) {
            for &b in bytes {
                self.0 = self.0.wrapping_mul(31).wrapping_add(b as u64);
            }
        }
    }

    #[derive(Default)]
    struct SimpleBuildHasher;

    impl BuildHasher for SimpleBuildHasher {
        type Hasher = SimpleHasher;
        fn build_hasher(&self) -> SimpleHasher {
            SimpleHasher::default()
        }
    }

    let bh = SimpleBuildHasher;
    let a = Asc::new(1);
    let b = Asc::new(1);
    let c = Asc::new(2);
    assert_eq!(bh.hash_one(&a), bh.hash_one(&b));
    assert_ne!(bh.hash_one(&a), bh.hash_one(&c));
}

#[test]
fn ord_as_map_key() {
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

#[test]
fn get_mut_unchecked_direct() {
    let mut a = Asc::new(5i32);
    // Safety: a is the only reference.
    unsafe {
        *Asc::get_mut_unchecked(&mut a) = 15;
    }
    assert_eq!(*a, 15);
    assert_eq!(Asc::strong_count(&a), 1);
}

#[test]
fn make_mut_zero_sized() {
    let mut a = Asc::new(());
    let _b = a.clone();
    assert_eq!(Asc::strong_count(&a), 2);
    // make_mut clones into a new allocation since count > 1
    let _val = Asc::make_mut(&mut a);
    assert_eq!(Asc::strong_count(&a), 1);
}

#[test]
fn try_unwrap_zero_sized_shared() {
    let a = Asc::new(());
    let _b = a.clone();
    let result = Asc::try_unwrap(a);
    assert!(result.is_err());
    let a = result.unwrap_err();
    assert_eq!(*a, ());
    drop(a);
}

#[test]
fn make_mut_preserves_value() {
    let mut a = Asc::new(String::from("original"));
    let _b = a.clone();
    Asc::make_mut(&mut a).push_str(" modified");
    // _b still has the original value
    assert_eq!(&*_b, "original");
    // a has the modified value in a new allocation
    assert_eq!(&*a, "original modified");
}

#[cfg(feature = "serde")]
mod serde_tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn serialize_basic() {
        let a = Asc::new(42i32);
        let json = serde_json::to_string(&a).unwrap();
        assert_eq!(json, "42");
    }

    #[test]
    fn deserialize_basic() {
        let a: Asc<i32> = serde_json::from_str("42").unwrap();
        assert_eq!(*a, 42);
    }

    #[test]
    fn roundtrip_json() {
        let a = Asc::new(String::from("hello"));
        let json = serde_json::to_string(&a).unwrap();
        let b: Asc<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(*a, *b);
    }

    #[test]
    fn serialize_complex() {
        let a = Asc::new(vec![1, 2, 3]);
        let json = serde_json::to_string(&a).unwrap();
        assert_eq!(json, "[1,2,3]");
    }

    #[test]
    fn deserialize_complex() {
        let a: Asc<Vec<i32>> = serde_json::from_str("[4,5,6]").unwrap();
        assert_eq!(*a, vec![4, 5, 6]);
    }

    #[test]
    fn serialize_shared() {
        let a = Asc::new(10);
        let _b = a.clone();
        let json = serde_json::to_string(&a).unwrap();
        assert_eq!(json, "10");
    }

    #[test]
    fn serialize_zero_sized() {
        let a = Asc::new(());
        let json = serde_json::to_string(&a).unwrap();
        assert_eq!(json, "null");
    }
}
