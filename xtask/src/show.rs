use crate::loose_shards::LooseShards;
use ggus::{
    GGufFileHeader, GGufMetaDataValueType, GGufMetaKV, GGufMetaKVPairs, GGufReadError, GGufReader,
    GGufTensorInfo, GGufTensors,
};
use std::{fmt, fs::File, path::PathBuf};

#[derive(Args, Default)]
pub struct ShowArgs {
    #[clap(long, short)]
    file: PathBuf,
    #[clap(long)]
    shards: bool,
}

const YES: &str = "✔️  ";
const ERR: &str = "❌  ";

struct Failed;

impl ShowArgs {
    pub fn show(self) {
        let files = if self.shards {
            LooseShards::from(&*self.file)
                .into_iter()
                .filter(|p| p.is_file())
                .collect::<Vec<_>>()
        } else if self.file.is_file() {
            vec![self.file]
        } else {
            vec![]
        };

        if files.is_empty() {
            println!("{ERR}No file found.");
            return;
        }

        for file in files {
            let file_name = file.file_name().unwrap().to_str().unwrap();
            println!(
                "\
+-{0:-<1$}-+
| {file_name} |
+-{0:-<1$}-+
",
                "",
                file_name.len()
            );

            let file = File::open(file).unwrap();
            let file = unsafe { memmap2::Mmap::map(&file) }.unwrap();

            let header = unsafe { file.as_ptr().cast::<GGufFileHeader>().read() };
            if let Err(Failed) = show_header(&header) {
                return;
            }

            let cursor = header.nbytes();
            let kvs = match GGufMetaKVPairs::scan(header.metadata_kv_count, &file[cursor..]) {
                Ok(kvs) => kvs,
                Err(e) => {
                    println!("{ERR}{e:?}");
                    return;
                }
            };
            if let Err(Failed) = show_meta_kvs(&kvs) {
                return;
            }

            let cursor = cursor + kvs.nbytes();
            let tensors = match GGufTensors::scan(header.tensor_count, &file[cursor..]) {
                Ok(tensors) => tensors,
                Err(e) => {
                    println!("{ERR}{e:?}");
                    return;
                }
            };
            if let Err(Failed) = show_tensors(&tensors) {
                return;
            }

            println!();
        }
    }
}

fn show_title(title: &str) {
    println!(
        "\
{title}
{0:=<1$}
",
        "",
        title.len()
    );
}

fn show_header(header: &GGufFileHeader) -> Result<(), Failed> {
    show_title("Header");

    if header.is_magic_correct() {
        println!("{YES}Magic   = {:?}", header.magic().unwrap());
    } else {
        println!("{ERR}Magic   = {:?}", header.magic());
        return Err(Failed);
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
        return Err(Failed);
    }
    if header.version == 3 {
        println!("{YES}Version = {}", header.version);
    } else {
        println!("{ERR}Version = {}", header.version);
        return Err(Failed);
    }
    println!("{YES}MetaKVs = {}", header.metadata_kv_count);
    println!("{YES}Tensors = {}", header.tensor_count);
    println!();
    Ok(())
}

fn show_meta_kvs(kvs: &GGufMetaKVPairs) -> Result<(), Failed> {
    show_title("Meta KV");

    let Some(width) = kvs.keys().map(|k| k.len()).max() else {
        return Ok(());
    };
    let mut topic = kvs
        .kvs()
        .filter(|k| k.key().starts_with("general."))
        .collect::<Vec<_>>();
    if !topic.is_empty() {
        topic.sort_unstable_by_key(GGufMetaKV::key);
        for kv in topic {
            show_meta_kv(kv, width)?;
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
            show_meta_kv(kv, width)?;
        }
        println!();
    }
    Ok(())
}

fn show_meta_kv(kv: GGufMetaKV, width: usize) -> Result<(), Failed> {
    let key = kv.key();
    let ty = kv.ty();
    let mut reader = kv.value_reader();
    let mut buf = String::new();
    match fmt_meta_val(&mut reader, ty, 1, &mut buf) {
        Ok(()) => {
            println!("{YES}{key:·<width$} {buf}");
            Ok(())
        }
        Err(e) => {
            println!("{ERR}{key:·<width$} {e:?}");
            Err(Failed)
        }
    }
}

fn fmt_meta_val<'a>(
    reader: &mut GGufReader<'a>,
    ty: GGufMetaDataValueType,
    len: usize,
    buf: &mut String,
) -> Result<(), GGufReadError<'a>> {
    struct MultiLines<'a>(&'a str);
    impl fmt::Display for MultiLines<'_> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            writeln!(f)?;
            writeln!(f, "   +--")?;
            for line in self.0.lines() {
                writeln!(f, "   | {}", line)?;
            }
            write!(f, "   +--")
        }
    }

    fn fmt_exp(e: f32) -> String {
        if e == 0. {
            String::from("0")
        } else if e.abs().log10().abs() > 3. {
            format!("{e:e}")
        } else {
            format!("{e}")
        }
    }

    match len {
        0 => buf.push_str("[]"),
        1 => {
            use GGufMetaDataValueType as T;
            match ty {
                T::U8 => buf.push_str(&reader.read::<u8>()?.to_string()),
                T::I8 => buf.push_str(&reader.read::<i8>()?.to_string()),
                T::U16 => buf.push_str(&reader.read::<u16>()?.to_string()),
                T::I16 => buf.push_str(&reader.read::<i16>()?.to_string()),
                T::U32 => buf.push_str(&reader.read::<u32>()?.to_string()),
                T::I32 => buf.push_str(&reader.read::<i32>()?.to_string()),
                T::U64 => buf.push_str(&reader.read::<u64>()?.to_string()),
                T::I64 => buf.push_str(&reader.read::<i64>()?.to_string()),
                T::F32 => buf.push_str(&fmt_exp(reader.read::<f32>()?)),
                T::F64 => buf.push_str(&fmt_exp(reader.read::<f64>()? as _)),
                T::Bool => buf.push(if reader.read()? { '√' } else { '×' }),
                T::String => {
                    let str = reader.read_str()?;
                    if str.lines().nth(1).is_some() {
                        buf.push_str(&format!("{}", MultiLines(str)));
                    } else {
                        buf.push_str(&format!("`{str}`"));
                    }
                }
                T::Array => {
                    let (ty, len) = reader.read_arr_header()?;
                    fmt_meta_val(reader, ty, len, buf)?;
                }
            }
        }
        _ if len <= 8 => {
            buf.push('[');
            for i in 0..len {
                if i > 0 {
                    buf.push_str(", ");
                }
                fmt_meta_val(reader, ty, 1, buf)?;
            }
            buf.push(']');
        }
        _ => {
            buf.push('[');
            for _ in 0..8 {
                fmt_meta_val(reader, ty, 1, buf)?;
                buf.push_str(", ");
            }
            buf.push_str(&format!("...({} more)]", len - 8));
        }
    }
    Ok(())
}

fn show_tensors(tensors: &GGufTensors) -> Result<(), Failed> {
    show_title("Tensors");

    let Some(name_width) = tensors.names().map(|k| k.len()).max() else {
        return Ok(());
    };
    let mut tensors = tensors.iter().collect::<Vec<_>>();
    tensors.sort_unstable_by_key(GGufTensorInfo::offset);
    let off_width = tensors.last().unwrap().offset().to_string().len() + 1;
    for t in tensors {
        println!(
            "{YES}{:·<name_width$} {:?} +{:<#0off_width$x} {:?}",
            t.name(),
            t.ggml_type(),
            t.offset(),
            t.shape()
        );
    }
    Ok(())
}
