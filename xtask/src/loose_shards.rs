use std::{
    ffi::OsString,
    path::{Path, PathBuf},
};

pub(crate) struct LooseShards {
    dir: PathBuf,
    prefix: String,
    extension: OsString,
    shards_count: usize,
    shards_format: usize,
}

impl LooseShards {
    #[inline]
    pub const fn count(&self) -> usize {
        self.shards_count
    }

    #[inline]
    pub fn single_file(&self) -> PathBuf {
        self.dir.join(&self.prefix).with_extension(&self.extension)
    }
}

impl From<&'_ Path> for LooseShards {
    fn from(file: &Path) -> Self {
        let dir = file.parent().unwrap().to_path_buf();
        let extension = file.extension().unwrap_or_default().to_os_string();

        let stem = file.file_stem().unwrap().to_str().unwrap();
        let shards = stem.split('-').rev().take(3).collect::<Vec<_>>();
        if let [shards, "of", index] = &*shards {
            if let Ok(n) = shards.parse() {
                if index.parse::<usize>().is_ok() {
                    return Self {
                        dir,
                        prefix: stem.rsplitn(4, '-').last().unwrap().to_string(),
                        extension,
                        shards_count: n,
                        shards_format: shards.len(),
                    };
                }
            }
        }
        Self {
            dir,
            prefix: stem.to_string(),
            extension,
            shards_count: 0,
            shards_format: 0,
        }
    }
}

impl<'a> IntoIterator for &'a LooseShards {
    type Item = PathBuf;
    type IntoIter = Iter<'a>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Iter(self, 0)
    }
}

pub(crate) struct Iter<'a>(&'a LooseShards, usize);

impl<'a> Iterator for Iter<'a> {
    type Item = PathBuf;

    fn next(&mut self) -> Option<Self::Item> {
        match self.0.shards_count {
            0 => {
                if self.1 == 0 {
                    self.1 += 1;
                    Some(self.0.single_file())
                } else {
                    None
                }
            }
            n => {
                if self.1 < n {
                    self.1 += 1;
                    Some(
                        self.0
                            .dir
                            .join(format!(
                                "{}-{:0fmt$}-of-{n:0fmt$}",
                                self.0.prefix,
                                self.1,
                                fmt = self.0.shards_format
                            ))
                            .with_extension(&self.0.extension),
                    )
                } else {
                    None
                }
            }
        }
    }
}
