use std::fmt;

pub fn print_file_info(n_tensors: usize, n_meta_kvs: usize, n_bytes: usize) {
    println!("   Number of tensors: {n_tensors}");
    println!("   Number of meta kvs: {n_meta_kvs}");
    println!("   File size: {}", MemSize(n_bytes));
    println!();
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(transparent)]
pub struct MemSize(pub usize);

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
