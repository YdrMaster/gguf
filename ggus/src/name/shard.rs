use fancy_regex::Regex;
use std::{fmt, num::NonZero, str::FromStr, sync::LazyLock};

#[derive(Clone, Debug)]
pub struct Shard {
    pub index: NonZero<u32>,
    pub count: NonZero<u32>,
}

impl Default for Shard {
    fn default() -> Self {
        Self {
            index: NonZero::new(1).unwrap(),
            count: NonZero::new(1).unwrap(),
        }
    }
}

impl FromStr for Shard {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        const PATTERN: &str = r"^(\d{5})-of-(\d{5})$";
        static REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(PATTERN).unwrap());

        let captures = REGEX.captures(s).unwrap().unwrap();
        Ok(Self {
            index: captures.get(1).unwrap().as_str().parse().unwrap(),
            count: captures.get(2).unwrap().as_str().parse().unwrap(),
        })
    }
}

impl fmt::Display for Shard {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let &Self { index, count } = self;
        write!(f, "{index:05}-of-{count:05}")
    }
}
