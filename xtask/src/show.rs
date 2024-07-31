use crate::{loose_shards::LooseShards, split_keys, ERR, YES};
use ggus::{
    GGufFileHeader, GGufMetaDataValueType, GGufMetaKV, GGufMetaKVPairs, GGufReadError, GGufReader,
    GGufTensors,
};
use std::{collections::HashSet, fmt, fs::File, path::PathBuf};

#[derive(Args, Default)]
pub struct ShowArgs {
    /// The file to show
    file: PathBuf,
    /// If set, show all shards in the directory
    #[clap(long)]
    shards: bool,
    /// How many elements to show in arrays, `all` for all elements
    #[clap(long, short = 'n', default_value = "8")]
    array_detail: String,
    /// Meta to show (split with `,`)
    #[clap(long, short = 'm')]
    filter_meta: Option<String>,
    /// Tensors to show (split with `,`)
    #[clap(long, short = 't')]
    filter_tensor: Option<String>,
}

struct Failed;

impl ShowArgs {
    pub fn show(self) {
        let Self {
            file,
            shards,
            array_detail,
            filter_meta,
            filter_tensor,
        } = self;

        let detail = match array_detail.trim().to_lowercase().as_str() {
            "all" => usize::MAX,
            s => s
                .parse()
                .expect("Invalid array detail, should be an integer or `all`"),
        };
        let filter_meta = split_keys(&filter_meta);
        let filter_tensor = split_keys(&filter_tensor);

        let files = if shards {
            LooseShards::from(&*file)
                .into_iter()
                .filter(|p| p.is_file())
                .collect::<Vec<_>>()
        } else if file.is_file() {
            vec![file]
        } else {
            vec![]
        };

        if files.is_empty() {
            eprintln!("{ERR}No file found.");
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
                println!();
                continue;
            }

            let cursor = header.nbytes();
            let kvs = match GGufMetaKVPairs::scan(header.metadata_kv_count, &file[cursor..]) {
                Ok(kvs) => kvs,
                Err(e) => {
                    eprintln!("{ERR}{e:?}");
                    println!();
                    continue;
                }
            };
            if show_meta_kvs(&kvs, &filter_meta, detail).is_err() {
                println!();
                continue;
            }

            let cursor = cursor + kvs.nbytes();
            let tensors = match GGufTensors::scan(header.tensor_count, &file[cursor..]) {
                Ok(tensors) => tensors,
                Err(e) => {
                    eprintln!("{ERR}{e:?}");
                    println!();
                    continue;
                }
            };
            let _ = show_tensors(&tensors, &filter_tensor);
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
        eprintln!("{ERR}Magic   = {:?}", header.magic());
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
        eprintln!("{ERR}Endian  = {native_endian}");
        return Err(Failed);
    }
    if header.version == 3 {
        println!("{YES}Version = {}", header.version);
    } else {
        eprintln!("{ERR}Version = {}", header.version);
        return Err(Failed);
    }
    println!("{YES}MetaKVs = {}", header.metadata_kv_count);
    println!("{YES}Tensors = {}", header.tensor_count);
    println!();
    Ok(())
}

fn show_meta_kvs<'a>(
    kvs: &GGufMetaKVPairs,
    filter: &Option<HashSet<&'a str>>,
    detail: usize,
) -> Result<(), Failed> {
    let kvs = filter.as_ref().map_or_else(
        || kvs.kvs().collect::<Vec<_>>(),
        |to_keep| {
            kvs.kvs()
                .filter(move |m| to_keep.contains(m.key()))
                .collect::<Vec<_>>()
        },
    );

    if let Some(width) = kvs.iter().map(|kv| kv.key().len()).max() {
        show_title("Meta KV");
        for kv in kvs {
            show_meta_kv(kv, width, detail)?;
        }
        println!();
    }

    Ok(())
}

fn show_meta_kv(kv: GGufMetaKV, width: usize, detail: usize) -> Result<(), Failed> {
    let key = kv.key();
    let ty = kv.ty();
    let mut reader = kv.value_reader();
    let mut buf = String::new();
    match fmt_meta_val(&mut reader, ty, 1, detail, &mut buf) {
        Ok(()) => {
            println!("{YES}{key:·<width$} {buf}");
            Ok(())
        }
        Err(e) => {
            eprintln!("{ERR}{key:·<width$} {e:?}");
            Err(Failed)
        }
    }
}

fn fmt_meta_val<'a>(
    reader: &mut GGufReader<'a>,
    ty: GGufMetaDataValueType,
    len: usize,
    detail: usize,
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
                    fmt_meta_val(reader, ty, len, detail, buf)?;
                }
            }
        }
        _ if len <= detail => {
            buf.push('[');
            for i in 0..len {
                if i > 0 {
                    buf.push_str(", ");
                }
                fmt_meta_val(reader, ty, 1, detail, buf)?;
            }
            buf.push(']');
        }
        _ => {
            buf.push('[');
            for _ in 0..detail {
                fmt_meta_val(reader, ty, 1, detail, buf)?;
                buf.push_str(", ");
            }
            buf.push_str(&format!("...({} more)]", len - detail));
        }
    }
    Ok(())
}

fn show_tensors<'a>(
    tensors: &GGufTensors,
    filter: &Option<HashSet<&'a str>>,
) -> Result<(), Failed> {
    let tensors = filter.as_ref().map_or_else(
        || tensors.iter().collect::<Vec<_>>(),
        |to_keep| {
            tensors
                .iter()
                .filter(move |t| to_keep.contains(t.name()))
                .collect::<Vec<_>>()
        },
    );

    if let Some(name_width) = tensors.iter().map(|t| t.name().len()).max() {
        show_title("Tensors");
        let tensors = tensors.iter().collect::<Vec<_>>();
        let off_width = tensors.last().unwrap().offset().to_string().len() + 1;
        for t in tensors {
            println!(
                "{YES}{:·<name_width$} {:?} +{:<#0off_width$x} {:?}",
                t.name(),
                t.ggml_type(),
                t.offset(),
                t.shape(),
            );
        }
    }

    Ok(())
}
