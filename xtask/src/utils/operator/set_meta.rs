use super::{super::MetaValue, Content, Operator};
use ggus::{GGufMetaDataValueType as Ty, GGufWriter, GENERAL_ALIGNMENT};
use internal::StrCollector;
use regex::Regex;
use std::{collections::HashMap, fmt::Debug, str::FromStr, sync::LazyLock};

impl Operator {
    #[inline]
    pub fn set_meta_by_cfg(cfg: &str) -> Self {
        let mut ans = HashMap::new();

        let mut state = None;
        for line in cfg.lines() {
            state = State::transfer(state, line, &mut ans);
        }
        if let Some(State::StrPedding(str) | State::StrAppending(str)) = state {
            str.save_to(&mut ans)
        }

        Self::SetMeta(ans)
    }
}

impl Content<'_> {
    pub(super) fn set_meta(&mut self, mut map: HashMap<String, (Ty, Vec<u8>)>) {
        for (k, v) in &mut self.meta_kvs {
            if let Some((ty, vec)) = map.remove(&**k) {
                // 禁止修改元信息类型
                assert_eq!(v.ty, ty);
                v.value = vec.into();
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
                // 空行维持当前状态
                if line.is_empty() {
                    return None;
                }

                macro_rules! regex {
                    ($name:ident $pattern:expr) => {
                        static $name: LazyLock<Regex> =
                            LazyLock::new(|| Regex::new($pattern).unwrap());
                    };
                }
                regex!(KV_REGEX  r"^`(?<Key>(\w+\.)*\w+)`\s*(?<Type>\S+)");
                regex!(ARR_REGEX r"^\[(\w+)\](\S+)?$");

                // 匹配元信息配置项
                let matches = KV_REGEX.captures(line).expect("Expect meta kv config item");
                let key = matches.name("Key").unwrap().as_str().to_string();
                let ty = matches.name("Type").unwrap();
                let val = line[ty.end()..].trim();
                let ty = ty.as_str();

                if let Some(sep) = ty.strip_prefix("str") {
                    // 配置字符串
                    if sep.is_empty() {
                        // 无分隔符，必须为单行字符串，以双引号包围
                        let val = val.strip_prefix('"').unwrap().strip_suffix('"').unwrap();
                        map.insert(key, write_val(Ty::String, val));
                        None
                    } else {
                        // 有分隔符，必须为多行字符串，内容为空
                        assert!(val.is_empty());
                        // 状态转移到字符串预备
                        Some(State::StrPedding(StrCollector::new(key, sep)))
                    }
                } else if let Some(arr) = ARR_REGEX.captures(ty) {
                    // TODO: 配置数组类型
                    todo!("arr: {}", &arr[0])
                } else {
                    // 配置代数类型
                    map.insert(key, write_val(parse_ty(ty), val));
                    None
                }
            }
            Some(Self::StrPedding(mut str)) => {
                // 字符串预备状态
                if line.is_empty() {
                    // 空行维持当前状态
                    Some(Self::StrPedding(str))
                } else if str.append(line) {
                    // 开始拼接多行字符串
                    Some(Self::StrAppending(str))
                } else {
                    // 字符串结束，写入元信息并重用当前行
                    str.save_to(map);
                    Self::transfer(None, line, map)
                }
            }
            Some(Self::StrAppending(mut str)) => {
                if line.is_empty() {
                    // 空行，字符串结束，写入元信息
                    str.save_to(map);
                    None
                } else if str.append(line) {
                    // 继续拼接多行字符串
                    Some(Self::StrAppending(str))
                } else {
                    // 字符串结束，写入元信息并重用当前行
                    str.save_to(map);
                    Self::transfer(None, line, map)
                }
            }
        }
    }
}

mod internal {
    use std::collections::HashMap;

    use super::{write_val, Ty};

    #[derive(Debug)]
    pub(super) struct StrCollector {
        key: String,
        sep: String,
        val: Option<String>,
    }

    impl StrCollector {
        #[inline]
        pub fn new(key: String, sep: &str) -> Self {
            Self {
                key,
                sep: format!("{sep} "),
                val: None,
            }
        }

        #[inline]
        pub fn append(&mut self, line: &str) -> bool {
            if let Some(line) = line.strip_prefix(&self.sep) {
                if let Some(val) = self.val.as_mut() {
                    val.push('\n');
                    val.push_str(line);
                } else {
                    self.val = Some(line.to_string());
                }
                true
            } else {
                false
            }
        }

        #[inline]
        pub fn save_to(self, map: &mut HashMap<String, (Ty, Vec<u8>)>) {
            map.insert(
                self.key,
                write_val(Ty::String, self.val.unwrap_or_default()),
            );
        }
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
