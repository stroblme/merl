//! Compile merl's C runtime as `make build` does, its `main` renamed `merl_main` for
//! `merl_sys::run` to call.

fn main() {
    println!("cargo:rerun-if-changed=runtime");
    cc::Build::new()
        .files(["runtime/merl.c", "runtime/libsais.c"])
        .define("main", "merl_main")
        .opt_level(3)
        .flag("-pthread")
        // Nor does `make build` ask for any.
        .warnings(false)
        .compile("merl");
}
