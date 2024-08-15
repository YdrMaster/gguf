use std::{
    cmp::max,
    path::{Path, PathBuf},
};

#[derive(Clone)]
pub(crate) struct Shards<'a> {
    pub dir: &'a Path,
    pub name: &'a str,
    pub index: usize,
    pub count: usize,
    pub format: usize,
}

impl<'a> From<&'a Path> for Shards<'a> {
    fn from(file: &'a Path) -> Self {
        let dir = file.parent().unwrap();
        let name = file.file_name().unwrap().to_str().unwrap();
        let Some((name, "gguf")) = name.rsplit_once('.') else {
            panic!()
        };
        match &*name.rsplitn(4, '-').collect::<Vec<_>>() {
            [count_, "of", index_, name] => {
                if let Ok(index) = index_.parse() {
                    if let Ok(count) = count_.parse() {
                        return Self {
                            dir,
                            name,
                            index,
                            count,
                            format: max(index_.len(), count_.len()),
                        };
                    }
                };
            }
            [..] => {}
        }
        Self {
            dir,
            name,
            index: 0,
            count: 1,
            format: 5,
        }
    }
}

impl Shards<'_> {
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

impl Iterator for Shards<'_> {
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
