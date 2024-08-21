use fancy_regex::Regex;
use std::{fmt, str::FromStr, sync::LazyLock};

#[derive(Clone, PartialEq, Debug)]
pub struct SizeLabel {
    e: u32,
    a: u32,
    b: u32,
    l: char,
}

impl FromStr for SizeLabel {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        const PATTERN: &str = r"^(\d+x)?(\d+)(\.\d+)?([QTBMK])$";
        static REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(PATTERN).unwrap());

        let captures = REGEX.captures(s).unwrap().unwrap();
        Ok(Self {
            e: captures.get(1).map_or(0, |m| {
                m.as_str().strip_suffix('x').unwrap().parse().unwrap()
            }),
            a: captures.get(2).unwrap().as_str().parse().unwrap(),
            b: captures.get(3).map_or(0, |m| {
                m.as_str().strip_prefix('.').unwrap().parse().unwrap()
            }),
            l: captures.get(4).unwrap().as_str().chars().next().unwrap(),
        })
    }
}

impl fmt::Display for SizeLabel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let &Self { e, a, b, l } = self;
        match e {
            0 => {}
            _ => write!(f, "{e}x")?,
        }
        match b {
            0 => write!(f, "{a}{l}"),
            _ => write!(f, "{a}.{b}{l}"),
        }
    }
}
