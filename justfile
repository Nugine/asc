dev:
    just fmt
    just lint
    just test
    just test-unstable
    just miri

fmt:
    cargo fmt

lint:
    cargo +nightly clippy --all-features -- -D warnings

test:
    cargo test --features std,serde

test-unstable:
    cargo +nightly test --all-features

miri:
    cargo +nightly miri test --all-features

coverage *ARGS:
    cargo llvm-cov --all-features --html {{ARGS}}

doc:
    RUSTDOCFLAGS="--cfg docsrs" cargo +nightly doc --open --no-deps --all-features

ci:
    cargo fmt --all --check
    just lint
    just test
    just test-unstable
    just miri
