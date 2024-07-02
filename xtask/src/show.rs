use ggus::{
    GGmlReadError, GGmlReader, GGufFileHeader, GGufMetaKV, GGufMetaKVPairs, GGufTensorInfo,
    GGufTensors,
};
use std::{fmt::Display, fs::File, path::PathBuf};

#[derive(Args, Default)]
pub struct ShowArgs {
    #[clap(long, short)]
    file: PathBuf,
}

impl ShowArgs {
    pub fn show(self) {
        if self.file.is_file() {
            let file = File::open(self.file).unwrap();
            let file = unsafe { memmap2::Mmap::map(&file) }.unwrap();
            show(&file);
        } else {
            todo!()
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
        f: impl FnOnce(GGmlReader) -> Result<T, GGmlReadError>,
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
        T::String => show(kv, width, |mut r| r.read_str().map(str::to_string)),

        T::Array => show(kv, width, |mut r| {
            let (ty, len) = r.read_arr_header()?;
            if len <= 8 {
                fn show<'a, T: Display>(
                    mut reader: GGmlReader<'a>,
                    len: usize,
                    mut f: impl FnMut(&mut GGmlReader<'a>) -> Result<T, GGmlReadError<'a>>,
                ) -> Result<String, GGmlReadError<'a>> {
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
                    T::String => show(r, len, |r| r.read_str().map(str::to_string)),
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
