use fancy_regex::Regex;
use std::{fmt, str::FromStr, sync::LazyLock};

#[derive(Clone, Debug)]
pub struct Version {
    major: u32,
    minor: u32,
}

impl FromStr for Version {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        const PATTERN: &str = r"^v(\d+)\.(\d+)$";
        static REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(PATTERN).unwrap());

        let captures = REGEX.captures(s).unwrap().unwrap();
        Ok(Self {
            major: captures.get(1).unwrap().as_str().parse().unwrap(),
            minor: captures.get(2).unwrap().as_str().parse().unwrap(),
        })
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let &Self { major, minor } = self;
        write!(f, "v{major}.{minor}")
    }
}
