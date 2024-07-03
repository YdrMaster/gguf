use core::fmt;
use std::str::FromStr;

#[derive(Clone, Debug)]
pub struct GGufFileName {
    model: String,
    version: Version,
    parameters: ParameterScale,
    encoding_scheme: String,
    shard: Shard,
}

#[derive(Clone, Debug)]
pub enum GGufFileNameError {
    InvalidNameExt,
    InvalidShard,
    EncodingSchemeNotExists,
    ParametersNotExists,
    InvalidParameters,
    InvalidVersion,
}

impl FromStr for GGufFileName {
    type Err = GGufFileNameError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let Some(s) = s.strip_suffix(".gguf") else {
            return Err(Self::Err::InvalidNameExt);
        };
        let s = s.split('-').collect::<Vec<_>>();
        // 解析分片信息
        let (s, shard) = if let [s @ .., index, "of", total] = &*s {
            let index = index.parse().map_err(|_| Self::Err::InvalidShard)?;
            let total = total.parse().map_err(|_| Self::Err::InvalidShard)?;
            (s, Shard { index, total })
        } else {
            (&*s, Shard::SINGLE)
        };
        // 解析编码方案
        let (s, encoding_scheme) = if let [s @ .., scheme] = s {
            (s, scheme.to_string())
        } else {
            return Err(Self::Err::EncodingSchemeNotExists);
        };
        // 解析参数量
        let (s, parameters) = if let [s @ .., params] = s {
            let (experts_count, params) = if let Some((experts, params)) = params.split_once('x') {
                (
                    experts.parse().map_err(|_| Self::Err::InvalidParameters)?,
                    params,
                )
            } else {
                (0, *params)
            };
            let (count, prefix) = match params.as_bytes() {
                [c @ .., b'Q'] => (c, ScalePrefix::Q),
                [c @ .., b'T'] => (c, ScalePrefix::T),
                [c @ .., b'B'] => (c, ScalePrefix::B),
                [c @ .., b'M'] => (c, ScalePrefix::M),
                [c @ .., b'K'] => (c, ScalePrefix::K),
                _ => return Err(Self::Err::InvalidParameters),
            };
            let count = std::str::from_utf8(count)
                .map_err(|_| Self::Err::InvalidParameters)?
                .parse()
                .map_err(|_| Self::Err::InvalidParameters)?;
            (
                s,
                ParameterScale {
                    experts_count,
                    count,
                    prefix,
                },
            )
        } else {
            return Err(Self::Err::ParametersNotExists);
        };
        // 解析版本
        let (s, version) = if let [head @ .., version] = s {
            if let Ok(version) = version.parse::<Version>() {
                (head, version)
            } else {
                (s, Version::PRERELEASE)
            }
        } else {
            return Err(Self::Err::InvalidVersion);
        };
        Ok(Self {
            model: s.join(" "),
            version,
            parameters,
            encoding_scheme,
            shard,
        })
    }
}

impl fmt::Display for GGufFileName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.model.replace(' ', "-"))?;
        if self.version != Version::PRERELEASE {
            write!(f, "-v{}.{}", self.version.major, self.version.minor)?;
        }
        write!(f, "-{}-{}", self.parameters, self.encoding_scheme)?;
        if self.shard != Shard::SINGLE {
            write!(f, "-{:05}-of-{:05}", self.shard.index, self.shard.total)?;
        }
        write!(f, ".gguf")
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Version {
    major: u32,
    minor: u32,
}

impl FromStr for Version {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (major, minor) = s.strip_prefix('v').ok_or(())?.split_once('.').ok_or(())?;
        Ok(Self {
            major: major.parse().map_err(|_| ())?,
            minor: minor.parse().map_err(|_| ())?,
        })
    }
}

impl Version {
    pub const PRERELEASE: Self = Self { major: 0, minor: 0 };
}

#[derive(Clone, PartialEq, Debug)]
pub struct ParameterScale {
    experts_count: u32,
    count: f32,
    prefix: ScalePrefix,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum ScalePrefix {
    Q,
    T,
    B,
    M,
    K,
}

impl fmt::Display for ParameterScale {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.experts_count != 0 {
            write!(f, "{}x", self.experts_count)?;
        }
        write!(f, "{}", self.count)?;
        match self.prefix {
            ScalePrefix::Q => write!(f, "Q"),
            ScalePrefix::T => write!(f, "T"),
            ScalePrefix::B => write!(f, "B"),
            ScalePrefix::M => write!(f, "M"),
            ScalePrefix::K => write!(f, "K"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Shard {
    index: u16,
    total: u16,
}

impl Shard {
    pub const SINGLE: Self = Self { index: 0, total: 1 };
}

#[test]
fn test_parse_filename() {
    assert!(matches!(
        "Mixtral-v0.1-8x7B-Q2_K".parse::<GGufFileName>(),
        Err(GGufFileNameError::InvalidNameExt)
    ));
    assert!(matches!(
        "Mixtral-v0.1-8x7B.gguf".parse::<GGufFileName>(),
        Err(GGufFileNameError::InvalidParameters)
    ));
    assert!(matches!(
        "Mixtral-v0.1-8x7H-Q2_K.gguf".parse::<GGufFileName>(),
        Err(GGufFileNameError::InvalidParameters)
    ));
    let Ok(name) = "Mixtral-8x7B-Q2_K.gguf".parse::<GGufFileName>() else {
        panic!()
    };
    assert_eq!(name.model, "Mixtral");
    assert_eq!(name.version, Version::PRERELEASE);
    assert_eq!(
        name.parameters,
        ParameterScale {
            experts_count: 8,
            count: 7.0,
            prefix: ScalePrefix::B,
        }
    );
    assert_eq!(name.encoding_scheme, "Q2_K");
}
