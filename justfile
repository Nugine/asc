dev:
    just fmt
    just lint
    just test
    just miri

fmt:
    cargo fmt

lint:
    cargo clippy --all-features

test:
    cargo test --all-features

miri:
    cargo +nightly miri test --all-features

doc:
    RUSTDOCFLAGS="--cfg docsrs" cargo +nightly doc --open --no-deps --all-features

ci:
    cargo fmt --all --check
    cargo clippy --all-features -- -D warnings
    just test
    just miri
