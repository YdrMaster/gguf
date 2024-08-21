use super::GGufFileName;
use std::path::Path;

#[derive(Clone)]
pub struct GGufShardPath<'a> {
    pub dir: &'a Path,
    pub name: GGufFileName<'a>,
}

#[derive(Clone, Debug)]
pub enum GGufShardParseError {
    InvalidPathFormat,
    UnknownFileKind(String),
}

impl<'a> TryFrom<&'a Path> for GGufShardPath<'a> {
    type Error = GGufShardParseError;

    fn try_from(value: &'a Path) -> Result<Self, Self::Error> {
        type E = GGufShardParseError;
        let dir = value.parent().ok_or(E::InvalidPathFormat)?;
        let name = value
            .file_name()
            .ok_or(E::InvalidPathFormat)?
            .to_str()
            .ok_or(E::InvalidPathFormat)?
            .strip_suffix(".gguf")
            .ok_or(E::InvalidPathFormat)?;
        match &*name.rsplitn(4, '-').collect::<Vec<_>>() {
            &[count_, "of", index_, name] => {
                if let Ok(index) = index_.parse() {
                    if let Ok(count) = count_.parse() {
                        return Ok(Self {
                            dir,
                            name: GGufFileName::new(name)?,
                            index,
                            count,
                            format: max(index_.len(), count_.len()),
                        });
                    }
                };
            }
            [..] => {}
        }
        Ok(Self {
            dir,
            name: GGufFileName::new(name)?,
            index: 0,
            count: 1,
            format: 5,
        })
    }
}

impl GGufShardPath<'_> {
    #[inline]
    pub fn to_single(&self) -> PathBuf {
        self.dir.join(format!("{}.gguf", self.name))
    }

    #[inline]
    pub fn iter_all(&self) -> Self {
        Self {
            index: 0,
            ..self.clone()
        }
    }
}

impl Iterator for GGufShardPath<'_> {
    type Item = PathBuf;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.count {
            let i = self.index;
            self.index += 1;

            Some(if self.count == 1 {
                self.to_single()
            } else {
                self.dir.join(format!(
                    "{name}-{i:0fmt$}-of-{n:0fmt$}.gguf",
                    name = self.name,
                    i = i + 1,
                    n = self.count,
                    fmt = self.format,
                ))
            })
        } else {
            None
        }
    }
}
