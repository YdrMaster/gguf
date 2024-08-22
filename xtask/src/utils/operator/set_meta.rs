use super::{super::MetaValue, Content, Operator};
use ggus::{GGufMetaDataValueType as Ty, GGufWriter, GENERAL_ALIGNMENT};
use regex::Regex;
use std::{collections::HashMap, fmt::Debug, str::FromStr, sync::LazyLock};

impl Operator {
    #[inline]
    pub fn set_meta_by_cfg(cfg: String) -> Self {
        let mut ans = HashMap::new();

        let mut state = None;
        for line in cfg.lines() {
            state = State::transfer(state, line, &mut ans);
        }
        match state {
            Some(State::StrPedding(_)) => panic!("Empty multi-line string"),
            Some(State::StrAppending(collector)) => {
                ans.insert(collector.key, write_val(Ty::String, collector.val));
            }
            None => {}
        }

        Self::SetMeta(ans)
    }
}

impl Content<'_> {
    pub fn set_meta(&mut self, mut map: HashMap<String, (Ty, Vec<u8>)>) {
        for (k, v) in &mut self.meta_kvs {
            if let Some((ty, vec)) = map.remove(&**k) {
                *v = MetaValue {
                    ty,
                    value: vec.into(),
                }
            }
        }
        for (k, (ty, vec)) in map {
            if k == GENERAL_ALIGNMENT {
                assert_eq!(ty, Ty::U32);
                let &[a, b, c, d] = &*vec else {
                    panic!("Invalid alignment value: {vec:?}");
                };
                self.alignment = u32::from_le_bytes([a, b, c, d]) as _;
            } else if k.starts_with("split.") {
                panic!("Split is not allowed: {k}");
            } else {
                self.meta_kvs.insert(
                    k.into(),
                    MetaValue {
                        ty,
                        value: vec.into(),
                    },
                );
            }
        }
    }
}

#[derive(Debug)]
enum State {
    StrPedding(StrCollector),
    StrAppending(StrCollector),
}

impl State {
    fn transfer(
        current: Option<Self>,
        line: &str,
        map: &mut HashMap<String, (Ty, Vec<u8>)>,
    ) -> Option<Self> {
        match current {
            None => {
                macro_rules! regex {
                    ($name:ident $pattern:expr) => {
                        static $name: LazyLock<Regex> =
                            LazyLock::new(|| Regex::new($pattern).unwrap());
                    };
                }

                regex!(REGEX      r"^`(?<Key>(\w+\.)*\w+)`\s*(?<Type>\S+)");
                regex!(STR_REGEX  r"^str(\S+)?$");
                regex!(ARR_REGEX  r"^\[(\w+)\](\S+)?$");

                let Some(matches) = REGEX.captures(line) else {
                    return None;
                };

                let key = matches.name("Key").unwrap().as_str().to_string();
                let ty = matches.name("Type").unwrap();
                let val = line[ty.end()..].trim();
                let ty = ty.as_str();

                if let Some(str) = STR_REGEX.captures(ty) {
                    let sep = &str[1];
                    if sep.is_empty() {
                        let val = val.strip_prefix('"').unwrap().strip_suffix('"').unwrap();
                        map.insert(key, write_val(Ty::String, val));
                        None
                    } else {
                        assert!(val.is_empty());
                        Some(State::StrPedding(StrCollector {
                            key,
                            sep: format!("{sep} "),
                            val: String::new(),
                        }))
                    }
                } else if let Some(arr) = ARR_REGEX.captures(ty) {
                    todo!("arr: {}", &arr[0])
                } else {
                    map.insert(key.into(), write_val(parse_ty(ty), val));
                    None
                }
            }
            Some(Self::StrPedding(collector)) => {
                if line.is_empty() {
                    Some(Self::StrPedding(collector))
                } else {
                    Some(Self::StrAppending(collector.append(line)))
                }
            }
            Some(Self::StrAppending(collector)) => {
                if line.is_empty() {
                    map.insert(collector.key, write_val(Ty::String, collector.val));
                    None
                } else {
                    Some(Self::StrAppending(collector.append(line)))
                }
            }
        }
    }
}

#[derive(Debug)]
struct StrCollector {
    key: String,
    sep: String,
    val: String,
}

impl StrCollector {
    pub fn append(mut self, line: &str) -> Self {
        let val = line
            .strip_prefix(&self.sep)
            .unwrap_or_else(|| panic!("Line must start with {}", self.sep));
        self.val.push('\n');
        self.val.push_str(val);
        self
    }
}

fn parse_ty(ty: &str) -> Ty {
    match ty {
        "u8" => Ty::U8,
        "i8" => Ty::I8,
        "u16" => Ty::U16,
        "i16" => Ty::I16,
        "u32" => Ty::U32,
        "i32" => Ty::I32,
        "f32" => Ty::F32,
        "u64" => Ty::U64,
        "i64" => Ty::I64,
        "f64" => Ty::F64,
        "bool" => Ty::Bool,
        "str" => Ty::String,
        "arr" => Ty::Array,
        _ => panic!("Unknown type: {ty}"),
    }
}

fn write_val(ty: Ty, val: impl AsRef<str>) -> (Ty, Vec<u8>) {
    let val = val.as_ref();

    #[inline]
    fn parse<T>(val: &str) -> T
    where
        T: FromStr,
        T::Err: Debug,
    {
        val.parse().unwrap()
    }

    let val = match ty {
        Ty::U8 => parse::<u8>(val).to_le_bytes().to_vec(),
        Ty::I8 => parse::<i8>(val).to_le_bytes().to_vec(),
        Ty::U16 => parse::<u16>(val).to_le_bytes().to_vec(),
        Ty::I16 => parse::<i16>(val).to_le_bytes().to_vec(),
        Ty::U32 => parse::<u32>(val).to_le_bytes().to_vec(),
        Ty::I32 => parse::<i32>(val).to_le_bytes().to_vec(),
        Ty::F32 => parse::<f32>(val).to_le_bytes().to_vec(),
        Ty::U64 => parse::<u64>(val).to_le_bytes().to_vec(),
        Ty::I64 => parse::<i64>(val).to_le_bytes().to_vec(),
        Ty::F64 => parse::<f64>(val).to_le_bytes().to_vec(),
        Ty::Bool => match val {
            "true" => vec![1],
            "false" => vec![0],
            _ => panic!("Invalid bool value: {}", val),
        },
        Ty::String => {
            let mut vec = Vec::with_capacity(val.len() + size_of::<u64>());
            GGufWriter::new(&mut vec).write_str(val).unwrap();
            vec
        }
        Ty::Array => unreachable!(),
    };

    (ty, val)
}
