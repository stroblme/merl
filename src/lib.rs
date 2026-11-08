//! merl's ghost-text server, its C runtime compiled in, for a Rust program to run as one of its
//! modes: [`run`] is what the `merl-rt` binary's `main` is.

use std::ffi::{CString, OsString, c_char, c_int};
use std::os::unix::ffi::OsStringExt;
use std::sync::atomic::{AtomicBool, Ordering};

unsafe extern "C" {
    /// `runtime/merl.c`'s `main`, renamed by `build.rs`.
    fn merl_main(argc: c_int, argv: *mut *mut c_char) -> c_int;
}

/// Run merl as `merl-rt` with these arguments, the program's name first, and return its exit
/// code: `--suggest` prints one answer and returns, and a bad argument prints the usage and
/// returns 2. As an LSP server on stdio it ends the process itself once its client says `exit` or
/// goes away.
///
/// merl keeps its state in globals, starts a thread and forks `git`, so it is the process's one
/// job: call it first thing in `main`, and once, as a second call panics.
pub fn run(args: impl IntoIterator<Item = OsString>) -> i32 {
    static RAN: AtomicBool = AtomicBool::new(false);
    assert!(
        !RAN.swap(true, Ordering::SeqCst),
        "merl runs once per process"
    );
    let args: Vec<CString> = args
        .into_iter()
        .map(|arg| CString::new(arg.into_vec()).expect("no argument holds a NUL byte"))
        .collect();
    let mut argv: Vec<*mut c_char> = args.iter().map(|arg| arg.as_ptr().cast_mut()).collect();
    argv.push(std::ptr::null_mut());
    // SAFETY: Rust's runtime ignores SIGPIPE, where a C program's does not and `git` inherits it
    // through merl's exec, so the default comes back first and a closed pipe ends merl as it ends
    // `merl-rt`. `argv` is NUL-terminated C strings, which outlive the call and are never written:
    // merl only reads them and keeps `--vault`'s, and the guard above makes this the only call.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
        merl_main(args.len() as c_int, argv.as_mut_ptr())
    }
}

#[cfg(test)]
mod tests {
    /// merl's own argument parsing answers, and only the first call runs it.
    #[test]
    fn a_bad_argument_gets_the_usage_and_a_second_run_panics() {
        assert_eq!(super::run(["merl-rt".into(), "--bogus".into()]), 2);
        assert!(std::panic::catch_unwind(|| super::run(["merl-rt".into()])).is_err());
    }
}
