use std::{cmp::max, fmt, path::PathBuf};

pub struct FileInfo {
    pub path: PathBuf,
    pub n_tensors: usize,
    pub n_meta_kvs: usize,
    pub n_bytes: usize,
}

pub fn show_file_info(file_info: &[FileInfo]) {
    if file_info.is_empty() {
        return;
    }

    const PATH: &str = "Path";
    const SIZE: &str = "Size";
    const META_KVS: &str = "Meta KVs";
    const TENSORS: &str = "Tensors";

    let mut max_path_len = PATH.len();
    let mut max_size_len = SIZE.len();
    let mut max_meta_kvs_len = META_KVS.len();
    let mut max_tensors_len = TENSORS.len();

    let mut path = Vec::with_capacity(file_info.len());
    let mut size = Vec::with_capacity(file_info.len());
    let mut meta_kvs = Vec::with_capacity(file_info.len());
    let mut tensors = Vec::with_capacity(file_info.len());

    for info in file_info {
        let path_ = info.path.display().to_string();
        let size_ = MemSize(info.n_bytes).to_string();
        let meta_kvs_ = info.n_meta_kvs.to_string();
        let tensors_ = info.n_tensors.to_string();

        max_path_len = max(max_path_len, path_.len());
        max_size_len = max(max_size_len, size_.len());
        max_meta_kvs_len = max(max_meta_kvs_len, meta_kvs_.len());
        max_tensors_len = max(max_tensors_len, tensors_.len());

        path.push(path_);
        size.push(size_);
        meta_kvs.push(meta_kvs_);
        tensors.push(tensors_);
    }

    let line = format!("+-{0:-<max_path_len$}-+-{0:-<max_size_len$}-+-{0:-<max_meta_kvs_len$}-+-{0:-<max_tensors_len$}-+", "");
    println!("{line}");
    println!("| {PATH:^max_path_len$} | {SIZE:^max_size_len$} | {META_KVS:^max_meta_kvs_len$} | {TENSORS:^max_tensors_len$} |");
    println!("{line}");
    for (p, s, m, t) in itertools::izip!(path, size, meta_kvs, tensors) {
        println!("| {p:<max_path_len$} | {s:>max_size_len$} | {m:>max_meta_kvs_len$} | {t:>max_tensors_len$} |");
    }
    println!("{line}");
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
struct MemSize(pub usize);

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
