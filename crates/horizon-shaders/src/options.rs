//! Switch options (spec 8.1), as far as program-enable expressions (3.3) need them: which are on
//! by default. A switch is `#define NAME` (on) or `//#define NAME` (off) with nothing after the
//! name but a comment, in any stage or included file of the pack. Packs keep their options in
//! included `.glsl` files (or files of any name a source includes) and test them elsewhere
//! (OQ-13), so the test the spec asks for (`#ifdef NAME` / `#ifndef NAME`) may be in any file
//! of the pack.

use std::collections::{BTreeMap, BTreeSet};

use crate::Pack;

const SOURCES: [&str; 8] = ["vsh", "fsh", "gsh", "csh", "tcs", "tes", "glsl", "inc"];

/// The switch options that are on by default.
pub fn switches_on(pack: &Pack) -> BTreeSet<String> {
    let (mut on, mut off, mut tested) = (BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
    for text in option_texts(pack) {
        for line in text.lines() {
            let line = line.trim();
            let (commented, rest) = match line.strip_prefix("//") {
                Some(rest) => (true, rest.trim_start()),
                None => (false, line),
            };
            let Some((directive, rest)) = rest
                .strip_prefix('#')
                .map(str::trim_start)
                .and_then(|r| r.split_once(|c: char| c.is_whitespace()))
            else {
                continue;
            };
            let mut words = rest.split_whitespace();
            let Some(name) = words.next() else { continue };
            match directive {
                "ifdef" | "ifndef" if !commented => _ = tested.insert(name.to_owned()),
                "define" if words.next().is_none_or(|w| w.starts_with("//")) => {
                    let set = if commented { &mut off } else { &mut on };
                    set.insert(name.to_owned());
                }
                _ => {}
            }
        }
    }
    // Defined both ways is ambiguous (8.1): it keeps its default only where it is on everywhere.
    on.retain(|name| tested.contains(name) && !off.contains(name));
    on
}

/// The value options and their defaults: `#define NAME VALUE // [A B C]`, a value and the list
/// of the values it may take (8.1).
pub fn value_options(pack: &Pack) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for text in option_texts(pack) {
        for line in text.lines() {
            let Some(rest) = line.trim().strip_prefix('#') else {
                continue;
            };
            let Some(rest) = rest.trim_start().strip_prefix("define") else {
                continue;
            };
            let Some((code, comment)) = rest.split_once("//") else {
                continue;
            };
            let mut words = code.split_whitespace();
            let (Some(name), Some(value), None) = (words.next(), words.next(), words.next()) else {
                continue;
            };
            if comment.contains('[') && comment.contains(']') {
                out.entry(name.to_owned())
                    .or_insert_with(|| value.to_owned());
            }
        }
    }
    out
}

/// `macros` with the pack's options at their defaults: the switches that are on, and the
/// value options. `shaders.properties` and the program switches test them (3.3, 8.1).
pub fn with_options(pack: &Pack, macros: &crate::Macros) -> crate::Macros {
    let mut out = macros.clone();
    for name in switches_on(pack) {
        out.define(name, "");
    }
    for (name, value) in value_options(pack) {
        out.define(name, value);
    }
    out
}

/// The texts of the pack's sources and of the files they include.
fn option_texts(pack: &Pack) -> Vec<String> {
    let files = pack.all_files();
    let source = |path: &str| SOURCES.contains(&path.rsplit_once('.').map_or("", |(_, e)| e));
    let texts: Vec<(&String, String)> = files
        .iter()
        .filter(|p| source(p))
        .filter_map(|p| Some((p, pack.read_text(p)?)))
        .collect();
    // Files of other names count when a source includes them (`#include "shaders.settings"`).
    let included: BTreeSet<&str> = texts
        .iter()
        .flat_map(|(_, t)| t.lines())
        .filter_map(|l| l.trim().strip_prefix("#include"))
        .map(|r| r.trim().trim_matches(|c| c == '"' || c == '<' || c == '>'))
        .map(|r| r.rsplit('/').next().unwrap_or(r))
        .collect();
    let extra: Vec<(&String, String)> = files
        .iter()
        .filter(|p| !source(p) && included.contains(p.rsplit('/').next().unwrap_or(p)))
        .filter_map(|p| Some((p, pack.read_text(p)?)))
        .collect();
    texts.into_iter().chain(extra).map(|(_, t)| t).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::tests::zip;

    #[test]
    fn switches_need_a_test_and_no_value() {
        let pack = Pack::from_zip_bytes(
            "P.zip",
            zip(&[
                (
                    "shaders/lib/settings.glsl",
                    "  #define AO // ambient occlusion\n//#define BLOOM\n#define LEVEL 2 //[1 2]\n\
                     #define SPLIT\n#define UNTESTED\n",
                ),
                (
                    "shaders/world0/deferred.fsh",
                    "#ifdef AO\n#endif\n#ifndef BLOOM\n#endif\n#ifdef SPLIT\n#endif\n#ifdef LEVEL\n#endif\n",
                ),
                ("shaders/world-1/deferred.fsh", "//#define SPLIT\n"),
                (
                    "shaders/final.fsh",
                    "#include \"shaders.settings\"\n#ifdef GODRAYS\n#endif\n",
                ),
                ("shaders/shaders.settings", "    #define GODRAYS //Toggle\n"),
                ("shaders/notes.settings", "#define NOT_INCLUDED\n#ifdef NOT_INCLUDED\n"),
                ("shaders/shaders.properties", "#define NOT_SOURCE\n#ifdef NOT_SOURCE\n"),
            ]),
        )
        .unwrap();
        assert_eq!(
            switches_on(&pack),
            BTreeSet::from(["AO".to_owned(), "GODRAYS".to_owned()])
        );
    }
}
