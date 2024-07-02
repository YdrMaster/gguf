use ggus::{
    GGufFileHeader, GGufMetaKV, GGufMetaKVPairs, GGufReadError, GGufReader, GGufTensorInfo,
    GGufTensors,
};
use std::{
    fmt::{self, Display},
    fs::File,
    path::PathBuf,
};

#[derive(Args, Default)]
pub struct ShowArgs {
    #[clap(long, short)]
    file: PathBuf,
    #[clap(long)]
    shards: bool,
}

impl ShowArgs {
    pub fn show(self) {
        if self.file.is_file() {
            let files = if self.shards {
                let stem = self
                    .file
                    .file_stem()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .split('-')
                    .collect::<Vec<_>>();
                if let [head @ .., i, "of", s] = &*stem {
                    let len = i.len();
                    assert_eq!(s.len(), len);
                    let head = head.join("-");
                    let ext = self.file.extension().unwrap().to_str().unwrap();
                    (1..=s.parse::<usize>().unwrap())
                        .map(|i| {
                            let file_name = format!("{head}-{i:0len$}-of-{s}.{ext}");
                            self.file.with_file_name(file_name)
                        })
                        .filter(|p| p.is_file())
                        .collect::<Vec<_>>()
                } else {
                    vec![self.file]
                }
            } else {
                vec![self.file]
            };

            for file in files {
                let file_name = file.file_name().unwrap().to_str().unwrap();
                println!("+{}+", "-".repeat(file_name.len() + 2));
                println!("| {} |", file_name);
                println!("+{}+", "-".repeat(file_name.len() + 2));
                println!();

                let file = File::open(file).unwrap();
                let file = unsafe { memmap2::Mmap::map(&file) }.unwrap();
                show(&file);

                println!();
            }
        } else {
            println!("{ERR}File not found: {}", self.file.display());
            exit();
        }
    }
}

fn show(file: &[u8]) {
    let header = unsafe { file.as_ptr().cast::<GGufFileHeader>().read() };
    show_header(&header);

    let cursor = header.nbytes();
    let kvs = GGufMetaKVPairs::scan(header.metadata_kv_count, &file[cursor..]).unwrap();
    show_meta_kvs(&kvs);

    let cursor = cursor + kvs.nbytes();
    let tensors = GGufTensors::scan(header.tensor_count, &file[cursor..]).unwrap();
    show_tensors(&tensors);
}

const YES: &str = "✔️  ";
const ERR: &str = "❌  ";
fn exit() -> ! {
    std::process::exit(1)
}

fn show_title(title: &str) {
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
    println!();
}

fn show_header(header: &GGufFileHeader) {
    show_title("Header");

    if header.is_magic_correct() {
        println!("{YES}Magic   = {:?}", header.magic().unwrap());
    } else {
        println!("{ERR}Magic   = {:?}", header.magic());
        exit();
    }
    let native_endian = if u16::from_le(1) == 1 {
        "Little"
    } else if u16::from_be(1) == 1 {
        "Big"
    } else {
        "Unknown"
    };
    if header.is_native_endian() {
        println!("{YES}Endian  = {native_endian}");
    } else {
        println!("{ERR}Endian  = {native_endian}");
        exit();
    }
    if header.version == 3 {
        println!("{YES}Version = {}", header.version);
    } else {
        println!("{ERR}Version = {}", header.version);
        exit()
    }
    println!("{YES}MetaKVs = {}", header.metadata_kv_count);
    println!("{YES}Tensors = {}", header.tensor_count);
    println!();
}

fn show_meta_kvs(kvs: &GGufMetaKVPairs) {
    show_title("Meta KV");

    let Some(width) = kvs.keys().map(|k| k.len()).max() else {
        return;
    };
    let mut topic = kvs
        .kvs()
        .filter(|k| k.key().starts_with("general."))
        .collect::<Vec<_>>();
    if !topic.is_empty() {
        topic.sort_unstable_by_key(GGufMetaKV::key);
        for kv in topic {
            show_meta_kv(kv, width);
        }
        println!();
    }

    let mut topic = kvs
        .kvs()
        .filter(|k| !k.key().starts_with("general."))
        .collect::<Vec<_>>();
    if !topic.is_empty() {
        topic.sort_unstable_by_key(GGufMetaKV::key);
        for kv in topic {
            show_meta_kv(kv, width);
        }
        println!();
    }
}

fn show_meta_kv(kv: GGufMetaKV, width: usize) {
    fn show<T: Display>(
        kv: GGufMetaKV,
        width: usize,
        f: impl FnOnce(GGufReader) -> Result<T, GGufReadError>,
    ) {
        let key = kv.key();
        match f(kv.value_reader()) {
            Ok(v) => {
                println!("{YES}{key:width$} {v}");
            }
            Err(e) => {
                println!("{ERR}{key:width$} {e:?}");
                exit();
            }
        }
    }

    fn show_str<'a>(reader: &mut GGufReader<'a>) -> Result<String, GGufReadError<'a>> {
        let str = reader.read_str()?;
        if str.lines().nth(1).is_some() {
            struct MultiLines<'a>(&'a str);
            impl fmt::Display for MultiLines<'_> {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    writeln!(f)?;
                    writeln!(f, "   +--")?;
                    for line in self.0.lines() {
                        writeln!(f, "   | {}", line)?;
                    }
                    write!(f, "   +--")
                }
            }
            Ok(format!("{}", MultiLines(str)))
        } else {
            Ok(str.to_string())
        }
    }

    use ggus::GGufMetaDataValueType as T;
    match kv.ty() {
        T::U8 => show(kv, width, |mut r| r.read::<u8>()),
        T::I8 => show(kv, width, |mut r| r.read::<i8>()),
        T::U16 => show(kv, width, |mut r| r.read::<u16>()),
        T::I16 => show(kv, width, |mut r| r.read::<i16>()),
        T::U32 => show(kv, width, |mut r| r.read::<u32>()),
        T::I32 => show(kv, width, |mut r| r.read::<i32>()),
        T::U64 => show(kv, width, |mut r| r.read::<u64>()),
        T::I64 => show(kv, width, |mut r| r.read::<i64>()),
        T::F32 => show(kv, width, |mut r| r.read::<f32>().map(|x| format!("{x:e}"))),
        T::F64 => show(kv, width, |mut r| r.read::<f64>().map(|x| format!("{x:e}"))),
        T::Bool => show(kv, width, |mut r| {
            r.read_bool().map(|b| if b { '√' } else { '×' })
        }),
        T::String => show(kv, width, |mut r| show_str(&mut r)),

        T::Array => show(kv, width, |mut r| {
            let (ty, len) = r.read_arr_header()?;
            if len <= 8 {
                fn show<'a, T: Display>(
                    mut reader: GGufReader<'a>,
                    len: usize,
                    mut f: impl FnMut(&mut GGufReader<'a>) -> Result<T, GGufReadError<'a>>,
                ) -> Result<String, GGufReadError<'a>> {
                    let mut ans = String::from("[");
                    for i in 0..len {
                        if i > 0 {
                            ans.push_str(", ");
                        }
                        ans.push_str(&format!("{}", f(&mut reader)?));
                    }
                    ans.push(']');
                    Ok(ans)
                }
                match ty {
                    T::U8 => show(r, len, |r| r.read::<u8>()),
                    T::I8 => show(r, len, |r| r.read::<i8>()),
                    T::U16 => show(r, len, |r| r.read::<u16>()),
                    T::I16 => show(r, len, |r| r.read::<i16>()),
                    T::U32 => show(r, len, |r| r.read::<u32>()),
                    T::I32 => show(r, len, |r| r.read::<i32>()),
                    T::U64 => show(r, len, |r| r.read::<u64>()),
                    T::I64 => show(r, len, |r| r.read::<i64>()),
                    T::F32 => show(r, len, |r| r.read::<f32>().map(|x| format!("{x:e}"))),
                    T::F64 => show(r, len, |r| r.read::<f64>().map(|x| format!("{x:e}"))),
                    T::Bool => show(r, len, |r| r.read_bool().map(|b| if b { '√' } else { '×' })),
                    T::String => show(r, len, show_str),
                    T::Array => todo!(),
                }
            } else {
                Ok(format!("[{ty:?}; {len}]"))
            }
        }),
    }
}

fn show_tensors(tensors: &GGufTensors) {
    show_title("Tensors");

    let Some(name_width) = tensors.names().map(|k| k.len()).max() else {
        return;
    };
    let mut tensors = tensors.iter().collect::<Vec<_>>();
    tensors.sort_unstable_by_key(GGufTensorInfo::offset);
    let off_width = tensors.last().unwrap().offset().to_string().len() + 1;
    for t in tensors {
        println!(
            "{YES}{:name_width$} {:?} +{:<#0off_width$x} {:?}",
            t.name(),
            t.ggml_type(),
            t.offset(),
            t.shape()
        );
    }
}
