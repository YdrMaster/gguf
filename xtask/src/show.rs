use crate::{utils::compile_patterns, LogArgs};
use ggus::{
    GGufFileHeader, GGufFileName, GGufMetaDataValueType, GGufMetaKV, GGufReadError, GGufReader,
};
use indexmap::IndexMap;
use memmap2::Mmap;
use regex::Regex;
use std::{
    fmt,
    fs::File,
    path::{Path, PathBuf},
};

const YES: &str = "✔️  ";
const ERR: &str = "❌  ";

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
    /// Meta to show
    #[clap(long, short = 'm', default_value = "*")]
    filter_meta: String,
    /// Tensors to show
    #[clap(long, short = 't', default_value = "*")]
    filter_tensor: String,

    #[clap(flatten)]
    log: LogArgs,
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
            log,
        } = self;
        log.init();

        let detail = match array_detail.trim().to_lowercase().as_str() {
            "all" => usize::MAX,
            s => s
                .parse()
                .expect("Invalid array detail, should be an integer or `all`"),
        };
        let filter_meta = compile_patterns(&filter_meta);
        let filter_tensor = compile_patterns(&filter_tensor);

        let files = if shards {
            let dir = file.parent().unwrap();
            GGufFileName::try_from(&*file)
                .unwrap()
                .iter_all()
                .map(|name| dir.join(name.to_string()))
                .collect::<Vec<_>>()
        } else {
            vec![file]
        };

        for path in files {
            let file = match File::open(&path) {
                Ok(f) => unsafe { Mmap::map(&f) }.unwrap(),
                Err(e) => {
                    println!("{ERR}Failed to open file: {e:?}");
                    continue;
                }
            };

            show_file_name(&path);

            let mut reader = GGufReader::new(&file);

            let header = match show_header(&mut reader) {
                Ok(header) => header,
                Err(Failed) => {
                    println!();
                    continue;
                }
            };

            match show_meta_kvs(
                &mut reader,
                header.metadata_kv_count as _,
                &filter_meta,
                detail,
            ) {
                Ok(a) => a,
                Err(Failed) => {
                    println!();
                    continue;
                }
            };

            if header.tensor_count > 0 {
                let _ = show_tensors(&mut reader, header.tensor_count as _, &filter_tensor);
            }
        }
    }
}

fn show_file_name(path: &Path) {
    let file_name = path.file_name().unwrap().to_str().unwrap();
    println!(
        "\
+-{0:-<1$}-+
| {file_name} |
+-{0:-<1$}-+
",
        "",
        file_name.len(),
    );
}

fn show_header(reader: &mut GGufReader) -> Result<GGufFileHeader, Failed> {
    show_title("Header");

    let header = match reader.read_header() {
        Ok(header) => header,
        Err(e) => {
            println!("{ERR} Failed to read header: {e:?}");
            return Err(Failed);
        }
    };

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
    Ok(header)
}

fn show_meta_kvs(
    reader: &mut GGufReader,
    count: usize,
    filter: &Regex,
    detail: usize,
) -> Result<(), Failed> {
    let mut width = 0;
    let mut meta_kvs = IndexMap::new();
    for _ in 0..count {
        let kv = match reader.read_meta_kv() {
            Ok(kv) => kv,
            Err(e) => {
                println!("{ERR}Failed to read meta kv: {e:?}");
                return Err(Failed);
            }
        };
        let k = kv.key();
        if meta_kvs.contains_key(k) {
            println!("{ERR}Duplicate meta key: {k}");
            return Err(Failed);
        }
        if filter.is_match(k) {
            width = k.len().max(width);
            meta_kvs.insert(k, kv);
        }
    }

    if !meta_kvs.is_empty() {
        show_title("Meta KV");
        for (_, kv) in meta_kvs {
            show_meta_kv(kv, width, detail)?;
        }
        println!();
    }

    Ok(())
}

fn show_tensors(reader: &mut GGufReader, count: usize, filter: &Regex) -> Result<(), Failed> {
    let mut name_width = 0;
    let mut off_width = 0;
    let mut tensors = IndexMap::new();
    for _ in 0..count {
        let tensor = match reader.read_tensor_meta() {
            Ok(t) => t,
            Err(e) => {
                println!("{ERR}Failed to read tensor: {e:?}");
                return Err(Failed);
            }
        };
        let name = tensor.name();
        if tensors.contains_key(name) {
            println!("{ERR}Duplicate tensor name: {name}");
            return Err(Failed);
        }
        if filter.is_match(name) {
            let info = tensor.to_info();
            name_width = name.len().max(name_width);
            off_width = info.offset().to_string().len().max(off_width);
            tensors.insert(name, info);
        }
    }

    name_width += 1;
    if !tensors.is_empty() {
        show_title("Tensors");
        for (name, info) in tensors {
            let ty = format!("{:?}", info.ty());
            println!(
                "{YES}{name:·<name_width$}{ty:·>6} +{:<#0off_width$x} {:?}",
                info.offset(),
                info.shape(),
            );
        }
    }

    Ok(())
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

fn show_meta_kv(kv: GGufMetaKV, width: usize, detail: usize) -> Result<(), Failed> {
    let key = kv.key();
    let ty = kv.ty();
    let mut reader = kv.value_reader();
    let mut buf = String::new();
    match fmt_meta_val(&mut reader, ty, 1, detail, &mut buf) {
        Ok(()) => {
            println!("{YES}{key:·<width$}{:·>5}: {buf}", ty.name());
            Ok(())
        }
        Err(e) => {
            println!("{ERR}{key:·<width$}{:·>5}: {e:?}", ty.name());
            Err(Failed)
        }
    }
}

fn fmt_meta_val(
    reader: &mut GGufReader,
    ty: GGufMetaDataValueType,
    len: usize,
    detail: usize,
    buf: &mut String,
) -> Result<(), GGufReadError> {
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
