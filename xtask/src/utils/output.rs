use std::{
    fmt,
    num::ParseIntError,
    path::PathBuf,
    str::{from_utf8, FromStr},
};

#[derive(Args, Default)]
pub(crate) struct OutputArgs {
    /// Output directory for converted files
    #[clap(long, short)]
    output_dir: Option<PathBuf>,
    /// Max count of tensors per shard
    #[clap(long, short = 't')]
    max_tensors: Option<usize>,
    /// Max size in bytes per shard
    #[clap(long, short = 's')]
    max_bytes: Option<String>,
    /// If set, the first shard will not contain any tensor
    #[clap(long, short)]
    no_tensor_first: bool,
}

pub(crate) struct OutputConfig {
    pub dir: Option<PathBuf>,
    pub shard_max_tensor_count: usize,
    pub shard_max_file_size: MemSize,
    pub shard_no_tensor_first: bool,
}

impl From<OutputArgs> for OutputConfig {
    fn from(args: OutputArgs) -> Self {
        Self {
            dir: args.output_dir,
            shard_max_tensor_count: args.max_tensors.unwrap_or(usize::MAX),
            shard_max_file_size: args.max_bytes.map_or(Default::default(), |s| {
                s.parse().expect("Invalid max bytes size")
            }),
            shard_no_tensor_first: args.no_tensor_first,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub(crate) struct MemSize(pub usize);

impl Default for MemSize {
    #[inline]
    fn default() -> Self {
        Self(usize::MAX)
    }
}

impl MemSize {
    #[inline]
    pub const fn nbytes(self) -> usize {
        self.0
    }
}

impl FromStr for MemSize {
    type Err = ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        #[inline]
        fn parse_size_num(num: &[u8], k: usize) -> Result<usize, ParseIntError> {
            from_utf8(num).unwrap().parse().map(|n: usize| n << k)
        }

        match s.trim().as_bytes() {
            [num @ .., b'G'] => parse_size_num(num, 30),
            [num @ .., b'M'] => parse_size_num(num, 20),
            [num @ .., b'K'] => parse_size_num(num, 10),
            num => parse_size_num(num, 0),
        }
        .map(Self)
    }
}

impl fmt::Display for MemSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 == 0 {
            write!(f, "0")
        } else {
            let zeros = self.0.trailing_zeros();
            let (num, unit) = if zeros >= 40 {
                (self.0 >> 40, "TiB")
            } else if zeros >= 30 {
                (self.0 >> 30, "GiB")
            } else if zeros >= 20 {
                (self.0 >> 20, "MiB")
            } else if zeros >= 10 {
                (self.0 >> 10, "KiB")
            } else {
                (self.0, "B")
            };
            let num = num.to_string();
            let num = num.chars().collect::<Vec<_>>();
            let first = match num.len() % 3 {
                0 => 3,
                n => n,
            };
            match &num[..first] {
                [a] => write!(f, "{a}")?,
                [a, b] => write!(f, "{a}{b}")?,
                [a, b, c] => write!(f, "{a}{b}{c}")?,
                _ => unreachable!(),
            }
            let mut num = &num[first..];
            loop {
                match num {
                    [a, b, c, tail @ ..] => {
                        write!(f, ",{a}{b}{c}")?;
                        num = tail;
                    }
                    [] => break,
                    _ => unreachable!(),
                }
            }
            write!(f, "{unit}")
        }
    }
}
