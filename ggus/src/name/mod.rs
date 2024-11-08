mod shard;
mod size_label;
mod r#type;
mod version;

use fancy_regex::{Captures, Regex};
use r#type::Type;
use shard::Shard;
use size_label::SizeLabel;
use std::{borrow::Cow, fmt, num::NonZero, path::Path, sync::LazyLock};
use version::Version;

#[derive(Clone, Debug)]
pub struct GGufFileName<'a> {
    pub base_name: Cow<'a, str>,
    pub size_label: Option<SizeLabel>,
    pub fine_tune: Option<Cow<'a, str>>,
    pub version: Option<Version>,
    pub encoding: Option<Cow<'a, str>>,
    pub type_: Type,
    pub shard: Shard,
}

#[derive(Debug)]
pub struct GGufShardParseError;

impl<'a> TryFrom<&'a str> for GGufFileName<'a> {
    type Error = GGufShardParseError;

    fn try_from(value: &'a str) -> Result<Self, Self::Error> {
        let captures = match_name(value);
        Ok(Self {
            base_name: captures.name("BaseName").unwrap().as_str().into(),
            size_label: captures
                .name("SizeLabel")
                .map(|m| m.as_str().parse().unwrap()),
            fine_tune: captures.name("FineTune").map(|m| m.as_str().into()),
            version: captures
                .name("Version")
                .map(|m| m.as_str().parse().unwrap()),
            encoding: captures.name("Encoding").map(|m| m.as_str().into()),
            type_: captures
                .name("Type")
                .map_or(Type::Default, |m| m.as_str().parse().unwrap()),
            shard: captures
                .name("Shard")
                .map_or(Default::default(), |m| m.as_str().parse().unwrap()),
        })
    }
}

impl<'a> TryFrom<&'a Path> for GGufFileName<'a> {
    type Error = GGufShardParseError;
    #[inline]
    fn try_from(value: &'a Path) -> Result<Self, Self::Error> {
        Self::try_from(value.file_name().unwrap().to_str().unwrap())
    }
}

impl GGufFileName<'_> {
    #[inline]
    pub fn shard_count(&self) -> usize {
        self.shard.count.get() as _
    }

    #[inline]
    pub fn into_single(self) -> Self {
        Self {
            shard: Default::default(),
            ..self
        }
    }

    #[inline]
    pub fn iter_all(self) -> Self {
        Self {
            shard: Shard {
                index: NonZero::new(1).unwrap(),
                ..self.shard
            },
            ..self
        }
    }

    #[inline]
    pub fn split_n(self, n: usize) -> Self {
        Self {
            shard: Shard {
                index: NonZero::new(1).unwrap(),
                count: NonZero::new(n as _).unwrap(),
            },
            ..self
        }
    }
}

impl Iterator for GGufFileName<'_> {
    type Item = Self;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.shard.index.get() <= self.shard.count.get() {
            let ans = self.clone();
            self.shard.index = self.shard.index.checked_add(1).unwrap();
            Some(ans)
        } else {
            None
        }
    }
}

impl fmt::Display for GGufFileName<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.base_name)?;
        if let Some(size_label) = &self.size_label {
            write!(f, "-{size_label}")?;
        }
        if let Some(fine_tune) = &self.fine_tune {
            write!(f, "-{fine_tune}")?;
        }
        if let Some(version) = &self.version {
            write!(f, "-{version}")?;
        }
        if let Some(encoding) = &self.encoding {
            write!(f, "-{encoding}")?;
        }
        f.write_str(match self.type_ {
            Type::Default => "",
            Type::LoRA => "-LoRA",
            Type::Vocab => "-vocab",
        })?;
        if self.shard.count.get() > 1 {
            write!(f, "-{}", self.shard)?;
        }
        write!(f, ".gguf")
    }
}

fn match_name(value: &str) -> Captures {
    // See: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#validating-above-naming-convention>
    const PATTERN: &str = r"^(?<BaseName>[A-Za-z0-9\s]*(?:(?:-(?:(?:[A-Za-z\s][A-Za-z0-9\s]*)|(?:[0-9\s]*)))*))-(?:(?<SizeLabel>(?:\d+x)?(?:\d+\.)?\d+[A-Za-z](?:-[A-Za-z]+(\d+\.)?\d+[A-Za-z]+)?)(?:-(?<FineTune>[A-Za-z0-9\s-]+))?)?-(?:(?<Version>v\d+(?:\.\d+)*))(?:-(?<Encoding>(?!LoRA|vocab)[\w_]+))?(?:-(?<Type>LoRA|vocab))?(?:-(?<Shard>\d{5}-of-\d{5}))?\.gguf$";
    static REGEX: LazyLock<Regex> = LazyLock::new(|| Regex::new(PATTERN).unwrap());

    REGEX.captures(value).unwrap().unwrap()
}

#[test]
fn test_name() {
    match_name("MiniCPM3-1B-sft-v0.0-F16.gguf");
}
