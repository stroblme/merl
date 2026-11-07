/* merl.c — ghost text for a markdown vault, spoken as LSP 3.18
 * `textDocument/inlineCompletion` on stdio.
 *
 * An ∞-gram over the vault: every note's bytes in one suffix array,
 * queried for the longest suffix of the typed prefix the vault has seen,
 * then walked forward while the vault agrees on the next byte. Pinned
 * byte for byte against merl/ngram.py (the specification) by
 * tests/test_runtime.py.
 *
 *   cc -O3 -pthread -o merl-rt runtime/merl.c runtime/libsais.c
 *   ./merl-rt --vault ~/notes                        # LSP server on stdio
 *   ./merl-rt --vault ~/notes --suggest "asks one"   # one answer, then exit
 *
 * The vault is `--vault`, else the client's first workspace folder, else
 * its rootUri (accent's rootUri is the nearest .git, not the vault).
 *
 * There are three tiers, all the same `Index`: the document being typed
 * in, the other open documents, and the vault. The longest match wins;
 * in the queried document the caret's own prefix is not one of its own
 * matches.
 *
 * The vault index is built on a thread of its own and swapped in when it
 * is ready, so neither `initialize` nor a `didSave` stalls the protocol:
 * requests are answered from the index that is live while the next one
 * is being built. The first build has nothing to answer from yet, so a
 * request arriving during it returns empty. libsais builds it in linear time --
 * 67 MB of notes in 2.4 s and 330 MB, twice that while a rebuild is in
 * flight; an open document is reindexed on every change.
 *
 * ponytail: still the whole index every time, one vault. Incremental
 * rebuild is the upgrade path; this only takes it off the critical path.
 */
#define JSMN_STATIC
#define JSMN_PARENT_LINKS
#include "jsmn.h"

#include "libsais.h"

#include <ctype.h>
#include <dirent.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* ---- ∞-gram constants. merl/ngram.py has the same numbers. Each can be
 * set from the build line with -DMIN_N=8 and friends, to sweep them. */
#ifndef MAXN
#define MAXN 64 /* bytes of prefix consulted */
#endif
#ifndef MAX_LEN
#define MAX_LEN 64 /* bytes suggested at most */
#endif
#ifndef MIN_N
#define MIN_N 6 /* shortest match worth continuing */
#endif
#ifndef MIN_COUNT
#define MIN_COUNT 2 /* occurrences a match needs */
#endif
#ifndef LONG_N
#define LONG_N 32 /* bytes of context past which one occurrence is enough */
#endif
#ifndef SHARE
#define SHARE 0.5 /* agreement the next byte needs */
#endif

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "merl: out of memory\n"); exit(1); }
    return p;
}

static void *xrealloc(void *p, size_t n) {
    p = realloc(p, n ? n : 1);
    if (!p) { fprintf(stderr, "merl: out of memory\n"); exit(1); }
    return p;
}

/* ================================================== text → suffix array */

/* Bytes and their suffix array, whether that is the whole vault or one open
 * document. Everything below queries an index through this and nothing else. */
typedef struct {
    const unsigned char *text;
    size_t len;
    const int32_t *sa;
} Index;

static void index_free(Index *ix) {
    free((void *)ix->text);
    free((void *)ix->sa);
    ix->text = NULL;
    ix->len = 0;
    ix->sa = NULL;
}

/* A suffix array over `len` bytes, in linear time (libsais); NULL when there is
 * nothing to index or too much of it. Order is full lexicographic rather than
 * merl.ngram's 128-byte key, which changes no answer: `find` and the walk use
 * each range as a set, and both orders group the suffixes sharing a prefix into
 * one contiguous range. */
static int32_t *build_sa(const unsigned char *text, size_t len) {
    if (len == 0) return NULL;
    if (len > (size_t)INT32_MAX) {
        fprintf(stderr, "merl: %zu bytes is more than the 32-bit suffix array holds\n", len);
        return NULL;
    }
    int32_t *sa = xmalloc(len * sizeof *sa);
    if (libsais(text, sa, (int32_t)len, 0, NULL) != 0) {
        fprintf(stderr, "merl: could not build the suffix array\n");
        free(sa);
        return NULL;
    }
    return sa;
}

/* ================================================== the vault */

static char *vault_root;

static Index vault_index;      /* live, read only by the main thread */
static Index next_index;       /* being built, written only by the builder thread */

/* Progress is reported only where the client asked for it (`window.workDoneProgress`
 * at initialize), under one token for the life of the process. */
#define PROGRESS_TOKEN "merl/index"
static int progress_ok;
static int progress_created;       /* the token has been registered with the client */

static pthread_t builder;
static int building;               /* a builder thread is running */
static int have_index;             /* an index has been built at least once */
static int pending;                /* a save arrived mid-build; build again after */
static volatile sig_atomic_t built; /* the builder is done and can be joined */

typedef struct { char **items; size_t n, cap; } Paths;

static void push_path(Paths *p, char *s) {
    if (p->n == p->cap) {
        p->cap = p->cap ? 2 * p->cap : 64;
        p->items = xrealloc(p->items, p->cap * sizeof *p->items);
    }
    p->items[p->n++] = s;
}

static char *join(const char *a, const char *sep, const char *b) {
    char *s = xmalloc(strlen(a) + strlen(sep) + strlen(b) + 1);
    sprintf(s, "%s%s%s", a, sep, b);
    return s;
}

/* Read a NUL-separated list of names from fd onto out, keeping the *.md ones. */
static void read_names(int fd, Paths *out) {
    char *buf = NULL;
    size_t len = 0, cap = 0;
    for (;;) {
        if (len + 4096 > cap) buf = xrealloc(buf, cap = cap ? 2 * cap : 65536);
        ssize_t n = read(fd, buf + len, cap - len);
        if (n <= 0) break;
        len += (size_t)n;
    }
    size_t start = 0;
    for (size_t i = 0; i <= len; i++) {
        if (i < len && buf[i]) continue;
        size_t n = i - start;
        if (n > 3 && memcmp(buf + i - 3, ".md", 3) == 0) {
            char *rel = xmalloc(n + 1);
            memcpy(rel, buf + start, n);
            rel[n] = 0;
            push_path(out, rel);
        }
        start = i + 1;
    }
    free(buf);
}

/* The vault's *.md as git sees them: tracked or untracked, never ignored.
 *
 * One subprocess, and git's own rules -- .gitignore, .git/info/exclude and the
 * user's global core.excludesFile -- instead of a second implementation of them
 * here. This is what accent does with the vault's git status, and it keeps a
 * `target/`, a `.venv` or a vendored tree of somebody else's markdown out of the
 * suggestions. Zero when there is no repository, no git, or nothing tracked, and
 * then the plain walk below answers instead.
 *
 * ponytail: submodules are not listed (git does not recurse into them), and a
 * vault that gitignores its own notes gets nothing and falls back to the walk. */
static int git_paths(Paths *out) {
    int fds[2];
    if (pipe(fds) != 0) return 0;
    pid_t pid = fork();
    if (pid < 0) { close(fds[0]); close(fds[1]); return 0; }
    if (pid == 0) {
        dup2(fds[1], STDOUT_FILENO);
        close(fds[0]);
        close(fds[1]);
        int null = open("/dev/null", O_WRONLY);
        if (null >= 0) { dup2(null, STDERR_FILENO); close(null); }
        /* -C, so the names come back relative to the vault rather than to a
         * repository root the vault may sit below. */
        execlp("git", "git", "-C", vault_root, "ls-files", "-z", "--cached",
               "--others", "--exclude-standard", (char *)NULL);
        _exit(127);
    }
    close(fds[1]);
    size_t before = out->n;
    read_names(fds[0], out);
    close(fds[0]);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) continue;
    int ok = WIFEXITED(status) && WEXITSTATUS(status) == 0 && out->n > before;
    if (!ok) {
        while (out->n > before) free(out->items[--out->n]);
    }
    return ok;
}

/* Every *.md below dir; dot-entries and symlinks skipped, like merl.dataset.vault. */
static void walk(const char *dir, const char *rel, Paths *out) {
    DIR *d = opendir(dir);
    if (!d) return;
    struct dirent *e;
    while ((e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        char *path = join(dir, "/", e->d_name);
        struct stat st;
        if (lstat(path, &st) == 0 && !S_ISLNK(st.st_mode)) {
            char *sub = join(rel, *rel ? "/" : "", e->d_name);
            size_t n = strlen(e->d_name);
            if (S_ISDIR(st.st_mode)) {
                walk(path, sub, out);
                free(sub);
            } else if (S_ISREG(st.st_mode) && n > 3 && strcmp(e->d_name + n - 3, ".md") == 0) {
                push_path(out, sub);
            } else {
                free(sub);
            }
        }
        free(path);
    }
    closedir(d);
}

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(char *const *)a, *(char *const *)b);
}

/* Defined next to the other senders, below. */
static void progress_begin(void);
static void progress_report(size_t files, int sorting);
static void progress_end(void);

static double seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

/* Walk the vault into next_index. Runs on the builder thread and touches
 * nothing the main thread reads. */
static void build_index(void) {
    index_free(&next_index);
    if (!vault_root) return;
    Paths paths = {0};
    if (!git_paths(&paths)) walk(vault_root, "", &paths);
    qsort(paths.items, paths.n, sizeof *paths.items, cmp_str);
    unsigned char *corpus = NULL;
    size_t len = 0, cap = 0;
    double reported = seconds();
    for (size_t k = 0; k < paths.n; k++) {
        char *path = join(vault_root, "/", paths.items[k]);
        FILE *f = fopen(path, "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long size = ftell(f);
            fseek(f, 0, SEEK_SET);
            if (size > 0) {
                if (len + size + 1 > cap) {
                    cap = 2 * (len + size + 1);
                    corpus = xrealloc(corpus, cap);
                }
                len += fread(corpus + len, 1, size, f);
            } else if (len + 1 > cap) {
                cap = 2 * (len + 1);
                corpus = xrealloc(corpus, cap);
            }
            corpus[len++] = '\n';
            fclose(f);
        }
        free(path);
        free(paths.items[k]);
        /* The count so far, at most ten times a second: short notes are read by
         * the thousand a second. */
        if (seconds() - reported >= 0.1) {
            progress_report(k + 1, 0);
            reported = seconds();
        }
    }
    /* The final count, and that the suffix array is next: on 67 MB of notes it
     * takes three times as long as reading them. */
    progress_report(paths.n, 1);
    free(paths.items);
    /* The doubling above overshoots by up to the corpus again, and the suffix
     * array is four times its size on top: give the slack back before asking
     * for that. */
    next_index.text = corpus = xrealloc(corpus, len);
    next_index.len = len;
    next_index.sa = build_sa(corpus, len);
    if (len && !next_index.sa) index_free(&next_index);
    fprintf(stderr, "merl: indexed %zu files, %zu bytes from %s\n", paths.n,
            next_index.len, vault_root);
}

static void *build_worker(void *unused) {
    (void)unused;
    build_index();
    /* Said here rather than at the swap: the swap happens on the main thread when
     * the next message arrives, and a client that is sitting still waiting to hear
     * the index is ready would never send one. Anything asking for a suggestion
     * swaps first (`reap_build`), so nothing is answered from the old index after
     * this. */
    progress_end();
    built = 1;
    return NULL;
}

/* Take the built index, freeing the one it replaces. Only the main thread calls
 * this, and only between messages, so nothing reads an index mid-swap. */
static void swap_in(void) {
    index_free(&vault_index);
    vault_index = next_index;
    next_index = (Index){0};
    have_index = 1;
}

static void start_build(void);

static void finish_build(void) {
    pthread_join(builder, NULL);
    building = 0;
    swap_in();
    /* A save that arrived mid-build starts the next one, and the client is still
     * told one `begin`...`end` for the whole run rather than one per pass. */
    if (pending) { pending = 0; start_build(); }
}

/* Start a build, or remember to start one when the running build is done: two
 * saves in a row are one rebuild of whatever is on disk by then. */
static void start_build(void) {
    if (building) { pending = 1; return; }
    built = 0;
    progress_begin();
    if (pthread_create(&builder, NULL, build_worker, NULL) != 0) {
        /* No thread to be had: build here rather than serve nothing. */
        build_index();
        swap_in();
        progress_end();
        return;
    }
    building = 1;
}

/* Swap in a build that has finished, if one has. Called between messages. */
static void reap_build(void) {
    if (building && built) finish_build();
}

/* ================================================== open documents */

/* An open document, with an index of its own so the note being typed in can
 * answer for itself before the vault is asked.
 *
 * ponytail: the whole document is reindexed on every change -- about a
 * millisecond per 100 KB, so a multi-megabyte note would feel it. */
typedef struct { char *uri; unsigned char *text; size_t len; int32_t *sa; } Doc;

static Doc *docs;
static size_t ndocs;
static int utf16; /* position encoding: UTF-16 code units, else UTF-8 bytes */

static Doc *doc_find(const char *uri) {
    for (size_t i = 0; i < ndocs; i++)
        if (strcmp(docs[i].uri, uri) == 0) return &docs[i];
    return NULL;
}

static void doc_set(char *uri, unsigned char *text, size_t len) {
    Doc *d = doc_find(uri);
    if (d) {
        free(d->text);
        free(d->sa);
        free(uri);
    } else {
        docs = xrealloc(docs, (ndocs + 1) * sizeof *docs);
        d = &docs[ndocs++];
        d->uri = uri;
    }
    d->text = text;
    d->len = len;
    d->sa = build_sa(text, len);
}

static void doc_close(const char *uri) {
    Doc *d = doc_find(uri);
    if (!d) return;
    free(d->uri);
    free(d->text);
    free(d->sa);
    *d = docs[--ndocs];
}

/* Byte offset of (line, character), clamped to the line and the document:
 * the editor's text may lag its caret. */
static size_t offset_of(const Doc *d, long line, long ch) {
    size_t i = 0;
    for (long l = 0; l < line && i < d->len;)
        if (d->text[i++] == '\n') l++;
    long units = 0;
    while (i < d->len && d->text[i] != '\n' && units < ch) {
        unsigned char b = d->text[i];
        size_t step = b < 0x80 ? 1 : b < 0xE0 ? 2 : b < 0xF0 ? 3 : 4;
        if (i + step > d->len) step = d->len - i;
        units += utf16 ? (b >= 0xF0 ? 2 : 1) : (long)step;
        i += step;
    }
    return i;
}

/* ================================================== the ∞-gram */

/* Sign of (suffix at idx, over pattern's length) vs pattern; a shorter
 * suffix that is a prefix of the pattern sorts before it, as bytes do. */
static int cmp_pat(const Index *ix, int32_t idx, const unsigned char *pat, size_t m) {
    size_t avail = ix->len - (size_t)idx;
    int c = memcmp(ix->text + idx, pat, avail < m ? avail : m);
    if (c) return c;
    return avail < m ? -1 : 0;
}

/* Narrow [lo, hi) to the suffixes starting with pat: bisect_left, bisect_right. */
static void find(const Index *ix, const unsigned char *pat, size_t m, size_t *lo, size_t *hi) {
    size_t a = *lo, b = *hi;
    while (a < b) {
        size_t mid = a + (b - a) / 2;
        if (cmp_pat(ix, ix->sa[mid], pat, m) < 0) a = mid + 1; else b = mid;
    }
    size_t first = a;
    b = *hi;
    while (a < b) {
        size_t mid = a + (b - a) / 2;
        if (cmp_pat(ix, ix->sa[mid], pat, m) <= 0) a = mid + 1; else b = mid;
    }
    *lo = first;
    *hi = a;
}

/* Length after dropping a trailing incomplete UTF-8 sequence (merl.ngram.trim_utf8). */
static size_t trim_utf8(const unsigned char *b, size_t n) {
    size_t p = n;
    while (p > 0 && n - p < 4 && (b[p - 1] & 0xC0) == 0x80) p--;
    if (p == 0) return n;
    unsigned char lead = b[--p];
    size_t need = lead < 0xC0 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : lead < 0xF8 ? 4 : 1;
    return n - p < need ? p : n;
}

/* Whether `count` occurrences of `n` bytes of context are worth copying.
 * Confidence is graded by length: text the index repeats, or -- once the agreed
 * context is long enough to be no coincidence -- text it holds once
 * (merl.ngram._enough). */
static int enough(size_t count, size_t n) {
    return count >= MIN_COUNT || (n >= LONG_N && count >= 1);
}

/* One index's answer to one prefix: the range [lo, hi) it matches and how many
 * bytes of it. `self` is the position of the caret's own prefix, which is the
 * query rather than an answer and is counted nowhere. */
typedef struct {
    Index ix;
    size_t n, lo, hi;
    long self; /* -1 unless the index is the document being typed in */
    int kind;  /* how the prefix was aligned (`aligned`) */
} Match;

/* Whether byte `b` separates words: neither alphanumeric nor part of a
 * multi-byte character. */
static int is_sep(unsigned char b) { return !(isalnum(b) || b >= 0x80); }

/* How the last `n` bytes of `prefix` may be matched: 1 when they start with a
 * separator, which is then part of the pattern; 2 when they start at a word, the
 * byte before them being a separator, so the pattern is the words alone and every
 * occurrence has to start a word too (`at_word`); 0 when they start inside a
 * word -- byte alignment alone finishes "he walk" from "she walked". Either way
 * the context is whole words plus the partial word at the caret, and 2 is what
 * lets `compute_` after a space match the vault's `(compute_`. The whole prefix
 * counts as 1 when it is the whole document rather than MAXN of it
 * (merl.ngram._aligned). */
static int aligned(const unsigned char *prefix, size_t plen, size_t n) {
    unsigned char b = prefix[plen - n];
    if (is_sep(b) || (n == plen && plen < MAXN)) return 1;
    return n < plen && is_sep(prefix[plen - n - 1]) ? 2 : 0;
}

/* Whether the suffix at `idx` starts a word: the byte before it, if any, is a
 * separator. */
static int at_word(const Index *ix, int32_t idx) {
    return idx == 0 || is_sep(ix->text[idx - 1]);
}

/* Keep `ix`'s longest match if it beats the one held; ties stay with the held
 * one, so the caller's order of candidates is its order of preference. `off` is
 * the caret's offset in `ix`, or -1 when the caret is not in it. */
static void consider(Match *best, Index ix, long off,
                     const unsigned char *prefix, size_t plen) {
    for (size_t n = plen; n >= MIN_N && n > best->n; n--) {
        int kind = aligned(prefix, plen, n);
        if (!kind) continue;
        size_t lo = 0, hi = ix.len;
        find(&ix, prefix + plen - n, n, &lo, &hi);
        size_t count = hi - lo;
        if (kind == 2) { /* only the occurrences that start a word */
            count = 0;
            for (size_t k = lo; k < hi; k++) count += at_word(&ix, ix.sa[k]);
        }
        if (off >= 0 && count) count--; /* the query's own suffix is in there */
        if (enough(count, n)) {
            *best = (Match){ix, n, lo, hi, off < 0 ? -1 : off - (long)n, kind};
            return;
        }
    }
}

/* The continuation of `prefix` up to the end of the sentence or of the line,
 * into `out` (MAX_LEN bytes); 0 when nothing has anything to say or the answer
 * would only repeat `after`, the text already following the caret.
 *
 * Three tiers, longest match first: the document being typed in (`self_doc`,
 * with the caret at `off`), the other open documents, then the vault. */
static size_t suggest(const Doc *self_doc, size_t off,
                      const unsigned char *prefix, size_t plen,
                      const unsigned char *after, size_t alen, unsigned char *out) {
    if (plen > MAXN) { prefix += plen - MAXN; plen = MAXN; }
    Match best = {{0}, 0, 0, 0, -1, 0};
    if (self_doc && self_doc->sa)
        consider(&best, (Index){self_doc->text, self_doc->len, self_doc->sa},
                 (long)off, prefix, plen);
    for (size_t k = 0; k < ndocs; k++)
        if (&docs[k] != self_doc && docs[k].sa)
            consider(&best, (Index){docs[k].text, docs[k].len, docs[k].sa}, -1, prefix, plen);
    if (vault_index.sa) consider(&best, vault_index, -1, prefix, plen);
    if (!best.n) return 0;

    const Index *ix = &best.ix;
    size_t n = best.n, lo = best.lo, hi = best.hi, len = 0;
    unsigned char pat[MAXN + MAX_LEN];
    memcpy(pat, prefix + plen - n, n);
    int whole = 0; /* a complete answer, rather than one cut at the budget */
    while (len < MAX_LEN) {
        uint32_t counts[256] = {0};
        size_t total = 0;
        for (size_t k = lo; k < hi; k++) {
            size_t j = (size_t)ix->sa[k];
            if ((long)j == best.self) continue;
            if (best.kind == 2 && !at_word(ix, ix->sa[k])) continue;
            total++;
            j += n + len;
            if (j < ix->len) counts[ix->text[j]]++;
        }
        int b = 0; /* highest count, lowest byte on a tie */
        for (int i = 1; i < 256; i++) if (counts[i] > counts[b]) b = i;
        size_t c = counts[b];
        /* A tie is not a majority: the byte has to hold more than SHARE. */
        if (!enough(c, n + len) || (double)c <= SHARE * (double)total) break;
        if (b == '\n') { whole = 1; break; } /* the line ends here, and so does this */
        pat[n + len++] = (unsigned char)b;
        /* And so does the sentence the caret was inside: whitespace right after
         * `.`, `!` or `?`, which may be the last byte of the prefix. */
        unsigned char prev = pat[n + len - 2];
        if (isspace(b) && (prev == '.' || prev == '!' || prev == '?')) { whole = 1; break; }
        find(ix, pat, n + len, &lo, &hi);
    }
    len = trim_utf8(pat + n, len);
    /* Stopped short of a whole answer, so the last word is half-copied: drop it
     * rather than suggest it. */
    if (!whole)
        while (len && !is_sep(pat[n + len - 1])) len--;
    size_t seen = 0;
    while (seen < len && isspace(pat[n + seen])) seen++;
    if (seen == len) return 0; /* nothing left, or nothing but whitespace */
    /* `after` is NULL under --suggest, and alen == 0 says so before it is read:
     * a suggestion is never empty by here, so no empty `after` starts with one. */
    size_t line = 0;
    while (line < alen && after[line] != '\n') line++;
    if (alen && line >= len && memcmp(after, pat + n, len) == 0) return 0;
    memcpy(out, pat + n, len);
    return len;
}

/* ================================================== JSON in and out */

static const char *js; /* the message being handled */
static jsmntok_t *toks;
static int ntoks;

static int parse(const char *body, size_t len) {
    static int cap = 0;
    if (!cap) { cap = 256; toks = xmalloc(cap * sizeof *toks); }
    for (;;) {
        jsmn_parser p;
        jsmn_init(&p);
        int r = jsmn_parse(&p, body, len, toks, cap);
        if (r != JSMN_ERROR_NOMEM) return r;
        cap *= 2; /* a didChange carries the whole document */
        toks = xrealloc(toks, cap * sizeof *toks);
    }
}

static int tok_is(int t, const char *s) {
    size_t n = toks[t].end - toks[t].start;
    return toks[t].type == JSMN_STRING && strlen(s) == n && memcmp(js + toks[t].start, s, n) == 0;
}

/* Whether token `t` is the literal `true`. */
static int tok_true(int t) {
    return t >= 0 && toks[t].type == JSMN_PRIMITIVE && toks[t].end - toks[t].start == 4
           && memcmp(js + toks[t].start, "true", 4) == 0;
}

/* Value token of `key` in object `obj`, or -1. */
static int get(int obj, const char *key) {
    if (obj < 0 || toks[obj].type != JSMN_OBJECT) return -1;
    for (int i = obj + 1; i + 1 < ntoks; i++)
        if (toks[i].parent == obj && tok_is(i, key)) return i + 1;
    return -1;
}

/* k-th element of array `arr` (k = -1: the last), or -1. */
static int nth(int arr, int k) {
    if (arr < 0 || toks[arr].type != JSMN_ARRAY) return -1;
    int found = -1, seen = 0;
    for (int i = arr + 1; i < ntoks; i++)
        if (toks[i].parent == arr) {
            if (seen++ == k) return i;
            found = i;
        }
    return k < 0 ? found : -1;
}

static long tok_long(int t) {
    return t < 0 ? 0 : strtol(js + toks[t].start, NULL, 10);
}

static unsigned hex4(const char *s) {
    unsigned v = 0;
    for (int i = 0; i < 4; i++) {
        char c = s[i];
        v = 16 * v + (unsigned)(c >= 'a' ? c - 'a' + 10 : c >= 'A' ? c - 'A' + 10 : c - '0');
    }
    return v;
}

static size_t put_utf8(unsigned char *o, unsigned cp) {
    if (cp < 0x80) { o[0] = cp; return 1; }
    if (cp < 0x800) { o[0] = 0xC0 | cp >> 6; o[1] = 0x80 | (cp & 0x3F); return 2; }
    if (cp < 0x10000) {
        o[0] = 0xE0 | cp >> 12; o[1] = 0x80 | ((cp >> 6) & 0x3F); o[2] = 0x80 | (cp & 0x3F);
        return 3;
    }
    o[0] = 0xF0 | cp >> 18; o[1] = 0x80 | ((cp >> 12) & 0x3F);
    o[2] = 0x80 | ((cp >> 6) & 0x3F); o[3] = 0x80 | (cp & 0x3F);
    return 4;
}

/* A string token's text, escapes undone (jsmn already checked their shape). */
static unsigned char *unescape(int t, size_t *outlen) {
    const char *s = js + toks[t].start, *e = js + toks[t].end;
    unsigned char *out = xmalloc(e - s + 1), *o = out;
    while (s < e) {
        if (*s != '\\') { *o++ = *s++; continue; }
        if (++s >= e) break;
        char c = *s++;
        switch (c) {
        case 'n': *o++ = '\n'; break;
        case 't': *o++ = '\t'; break;
        case 'r': *o++ = '\r'; break;
        case 'b': *o++ = '\b'; break;
        case 'f': *o++ = '\f'; break;
        case 'u': {
            unsigned cp = hex4(s);
            s += 4;
            if (cp >= 0xD800 && cp < 0xDC00 && e - s >= 6 && s[0] == '\\' && s[1] == 'u') {
                unsigned lo = hex4(s + 2);
                if (lo >= 0xDC00 && lo < 0xE000) {
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    s += 6;
                }
            }
            o += put_utf8(o, cp);
            break;
        }
        default: *o++ = c; /* \" \\ \/ */
        }
    }
    *o = 0;
    if (outlen) *outlen = o - out;
    return out;
}

/* file:///a%20b → /a b */
static char *uri_to_path(const unsigned char *uri) {
    const char *s = (const char *)uri;
    if (strncmp(s, "file://", 7) == 0) s += 7;
    char *out = xmalloc(strlen(s) + 1), *o = out;
    while (*s) {
        if (*s == '%' && s[1] && s[2]) {
            char hex[3] = {s[1], s[2], 0};
            *o++ = (char)strtol(hex, NULL, 16);
            s += 3;
        } else {
            *o++ = *s++;
        }
    }
    *o = 0;
    return out;
}

typedef struct { char *p; size_t n, cap; } Buf;

static void buf_add(Buf *b, const char *s, size_t n) {
    if (b->n + n + 1 > b->cap) {
        b->cap = 2 * (b->n + n + 1);
        b->p = xrealloc(b->p, b->cap);
    }
    memcpy(b->p + b->n, s, n);
    b->n += n;
    b->p[b->n] = 0;
}

static void buf_fmt(Buf *b, const char *fmt, ...) {
    char s[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(s, sizeof s, fmt, ap);
    va_end(ap);
    buf_add(b, s, n < (int)sizeof s ? (size_t)n : sizeof s - 1);
}

static void buf_json_str(Buf *b, const unsigned char *s, size_t n) {
    buf_add(b, "\"", 1);
    for (size_t i = 0; i < n; i++) {
        unsigned char c = s[i];
        if (c == '"') buf_add(b, "\\\"", 2);
        else if (c == '\\') buf_add(b, "\\\\", 2);
        else if (c == '\n') buf_add(b, "\\n", 2);
        else if (c == '\r') buf_add(b, "\\r", 2);
        else if (c == '\t') buf_add(b, "\\t", 2);
        else if (c < 0x20) buf_fmt(b, "\\u%04x", c);
        else buf_add(b, (const char *)&c, 1);
    }
    buf_add(b, "\"", 1);
}

/* The builder thread reports its own progress, so two threads write here. One
 * message must not land inside another. */
static pthread_mutex_t out_lock = PTHREAD_MUTEX_INITIALIZER;

static void send(Buf *b) {
    pthread_mutex_lock(&out_lock);
    printf("Content-Length: %zu\r\n\r\n", b->n);
    fwrite(b->p, 1, b->n, stdout);
    fflush(stdout);
    pthread_mutex_unlock(&out_lock);
    free(b->p);
    b->p = NULL;
    b->n = b->cap = 0;
}

/* The request's id, verbatim (a number or a quoted string). */
static void id_of(int id, const char **p, size_t *n) {
    int quoted = toks[id].type == JSMN_STRING;
    *p = js + toks[id].start - quoted;
    *n = toks[id].end - toks[id].start + 2 * quoted;
}

static void reply(int id, const char *result) {
    const char *p;
    size_t n;
    id_of(id, &p, &n);
    Buf b = {0};
    buf_add(&b, "{\"jsonrpc\":\"2.0\",\"id\":", 22);
    buf_add(&b, p, n);
    buf_add(&b, ",\"result\":", 10);
    buf_add(&b, result, strlen(result));
    buf_add(&b, "}", 1);
    send(&b);
}

static void reply_error(int id, int code, const char *message) {
    const char *p;
    size_t n;
    id_of(id, &p, &n);
    Buf b = {0};
    buf_add(&b, "{\"jsonrpc\":\"2.0\",\"id\":", 22);
    buf_add(&b, p, n);
    buf_fmt(&b, ",\"error\":{\"code\":%d,\"message\":\"%s\"}}", code, message);
    send(&b);
}

/* A notification: no id, no answer expected. */
static void notify(const char *method, const char *params) {
    Buf b = {0};
    buf_fmt(&b, "{\"jsonrpc\":\"2.0\",\"method\":\"%s\",\"params\":%s}", method, params);
    send(&b);
}

/* Say the index is being rebuilt, so an editor can show it. `window/workDoneProgress/create`
 * registers the token; its answer carries no id this server remembers and is dropped by the
 * loop below, like any message with no method. */
static void progress_begin(void) {
    if (!progress_ok) return;
    if (!progress_created) {
        progress_created = 1;
        Buf b = {0};
        buf_fmt(&b, "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"window/workDoneProgress/create\","
                    "\"params\":{\"token\":\"%s\"}}", PROGRESS_TOKEN);
        send(&b);
    }
    notify("$/progress", "{\"token\":\"" PROGRESS_TOKEN "\",\"value\":{\"kind\":\"begin\","
                        "\"title\":\"Indexing\",\"cancellable\":false}}");
}

/* How far the build is, as a message an editor can show as it stands: "1 file",
 * "1200 files", digits ungrouped, and "1200 files, sorting" once they are all
 * read. Called from the builder thread, like `progress_end`. */
static void progress_report(size_t files, int sorting) {
    if (!progress_ok) return;
    char params[160];
    snprintf(params, sizeof params, "{\"token\":\"" PROGRESS_TOKEN "\",\"value\":{\"kind\":"
             "\"report\",\"message\":\"%zu file%s%s\"}}", files, files == 1 ? "" : "s",
             sorting ? ", sorting" : "");
    notify("$/progress", params);
}

/* Called from the builder thread; `send` is what makes that safe. */
static void progress_end(void) {
    if (!progress_ok) return;
    notify("$/progress", "{\"token\":\"" PROGRESS_TOKEN "\",\"value\":{\"kind\":\"end\"}}");
}

/* ================================================== the LSP loop */

static int read_message(char **body, size_t *len) {
    char line[512];
    long clen = -1;
    while (fgets(line, sizeof line, stdin)) {
        if (strcmp(line, "\r\n") == 0 || strcmp(line, "\n") == 0) break;
        if (strncasecmp(line, "Content-Length:", 15) == 0) clen = atol(line + 15);
    }
    if (clen < 0) return 0;
    *body = xmalloc(clen + 1);
    if (fread(*body, 1, clen, stdin) != (size_t)clen) return 0;
    (*body)[clen] = 0;
    *len = clen;
    return 1;
}

static unsigned char *text_of(int t, size_t *len) {
    if (t < 0 || toks[t].type != JSMN_STRING) { *len = 0; return xmalloc(1); }
    return unescape(t, len);
}

static void handle_initialize(int id, int params) {
    if (!vault_root) {
        int uri = get(nth(get(params, "workspaceFolders"), 0), "uri");
        if (uri < 0) uri = get(params, "rootUri");
        if (uri >= 0 && toks[uri].type == JSMN_STRING) {
            unsigned char *s = unescape(uri, NULL);
            vault_root = uri_to_path(s);
            free(s);
        }
    }
    int caps = get(params, "capabilities");
    utf16 = 1;
    int encodings = get(get(caps, "general"), "positionEncodings");
    for (int k = 0, t; (t = nth(encodings, k)) >= 0; k++)
        if (tok_is(t, "utf-8")) utf16 = 0;
    /* Only where the client said it can show one. An editor that did not ask hears nothing. */
    progress_ok = tok_true(get(get(caps, "window"), "workDoneProgress"));
    if (!vault_root) fprintf(stderr, "merl: no vault given, nothing to suggest\n");
    Buf b = {0};
    buf_fmt(&b, "{\"capabilities\":{\"positionEncoding\":\"%s\",\"textDocumentSync\":1,"
                "\"inlineCompletionProvider\":{}},\"serverInfo\":{\"name\":\"merl\","
                "\"version\":\"0.1.0\"}}", utf16 ? "utf-16" : "utf-8");
    reply(id, b.p);
    free(b.p);
    /* After the answer, because nothing may precede it on the wire; started rather than
     * waited for, so an editor opening a vault does not sit through the walk. */
    start_build();
}

static void handle_inline(int id, int params) {
    /* Keep serving shutdown while the first index is being built. */
    if (!have_index) {
        reply(id, "{\"items\":[]}");
        return;
    }
    int uri = get(get(params, "textDocument"), "uri");
    int pos = get(params, "position");
    unsigned char *u = text_of(uri, &(size_t){0});
    Doc *d = doc_find((char *)u);
    free(u);
    unsigned char out[MAX_LEN];
    size_t n = 0;
    if (d) {
        size_t off = offset_of(d, tok_long(get(pos, "line")), tok_long(get(pos, "character")));
        size_t start = off > MAXN ? off - MAXN : 0;
        n = suggest(d, off, d->text + start, off - start, d->text + off, d->len - off, out);
    }
    Buf b = {0};
    buf_add(&b, "{\"items\":[", 10);
    if (n) {
        buf_add(&b, "{\"insertText\":", 14);
        buf_json_str(&b, out, n);
        buf_add(&b, "}", 1);
    }
    buf_add(&b, "]}", 2);
    reply(id, b.p);
    free(b.p);
}

static void serve(void) {
    int initialized = 0, shut = 0;
    char *body;
    size_t len;
    while (read_message(&body, &len)) {
        reap_build();
        js = body;
        ntoks = parse(body, len);
        int method = ntoks > 0 ? get(0, "method") : -1;
        int id = ntoks > 0 ? get(0, "id") : -1;
        int params = ntoks > 0 ? get(0, "params") : -1;
        if (method < 0) {
            /* not a request or notification: nothing to do */
        } else if (tok_is(method, "initialize")) {
            handle_initialize(id, params);
            initialized = 1;
        } else if (tok_is(method, "exit")) {
            exit(shut ? 0 : 1);
        } else if (!initialized) {
            if (id >= 0) reply_error(id, -32002, "server not initialized");
        } else if (tok_is(method, "shutdown")) {
            shut = 1;
            reply(id, "null");
        } else if (tok_is(method, "textDocument/didOpen")) {
            int td = get(params, "textDocument");
            size_t n;
            unsigned char *text = text_of(get(td, "text"), &n);
            doc_set((char *)text_of(get(td, "uri"), &(size_t){0}), text, n);
        } else if (tok_is(method, "textDocument/didChange")) {
            int td = get(params, "textDocument");
            size_t n;
            unsigned char *text = text_of(get(nth(get(params, "contentChanges"), -1), "text"), &n);
            doc_set((char *)text_of(get(td, "uri"), &(size_t){0}), text, n);
        } else if (tok_is(method, "textDocument/didClose")) {
            unsigned char *u = text_of(get(get(params, "textDocument"), "uri"), &(size_t){0});
            doc_close((char *)u);
            free(u);
        } else if (tok_is(method, "textDocument/didSave")) {
            start_build();
        } else if (tok_is(method, "textDocument/inlineCompletion")) {
            handle_inline(id, params);
        } else if (id >= 0) {
            reply_error(id, -32601, "method not found");
        }
        free(body);
    }
    exit(shut ? 0 : 1);
}

int main(int argc, char **argv) {
    const char *one = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--vault") == 0 && i + 1 < argc) vault_root = argv[++i];
        else if (strcmp(argv[i], "--suggest") == 0 && i + 1 < argc) one = argv[++i];
        else {
            fprintf(stderr, "usage: %s [--vault DIR] [--suggest PREFIX]\n", argv[0]);
            return 2;
        }
    }
    if (one) {
        build_index();
        swap_in();
        unsigned char out[MAX_LEN];
        size_t n = suggest(NULL, 0, (const unsigned char *)one, strlen(one), NULL, 0, out);
        fwrite(out, 1, n, stdout);
        putchar('\n');
        return 0;
    }
    serve();
    return 0;
}
