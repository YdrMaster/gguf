use regex::Regex;
use std::{borrow::Cow, fmt, sync::OnceLock};

#[inline]
pub(crate) fn compile_patterns(patterns: &str) -> Regex {
    Regex::new(&format!("{}", Patterns(patterns))).unwrap()
}

struct Patterns<'a>(&'a str);

impl fmt::Display for Patterns<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        static REGEX: OnceLock<Regex> = OnceLock::new();
        // 匹配任何标识符、点、星号的组合
        let patterns = REGEX
            .get_or_init(|| Regex::new(r"[\w*.]+").unwrap())
            .find_iter(self.0);

        let mut patterns = patterns.into_iter();
        if let Some(pattern) = patterns.next() {
            write!(f, "{}", Pattern(pattern.as_str()))?;
        }
        for pattern in patterns {
            write!(f, "|{}", Pattern(pattern.as_str()))?;
        }
        Ok(())
    }
}

struct Pattern<'a>(&'a str);

impl fmt::Display for Pattern<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut iter = self.0.split('.').map(|s| {
            if s.is_empty() {
                Cow::Borrowed(r"\w+")
            } else if s.chars().all(|c| c == '*') {
                Cow::Borrowed(r"(\w+\.)*\w+")
            } else {
                static REGEX: OnceLock<Regex> = OnceLock::new();
                // 消除任何连续 *
                REGEX
                    .get_or_init(|| Regex::new(r"\*+").unwrap())
                    .replace_all(s, r"\w*")
            }
        });

        write!(f, "^")?;
        if let Some(ele) = iter.next() {
            write!(f, "{ele}")?;
        }
        for ele in iter {
            write!(f, r"\.{ele}")?;
        }
        write!(f, "$")
    }
}
