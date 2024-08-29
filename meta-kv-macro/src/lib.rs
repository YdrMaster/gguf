#![deny(warnings)]

mod ident_collector;

use ident_collector::IdentCollector;
use proc_macro::TokenStream;
use std::collections::HashSet;

#[proc_macro]
pub fn meta_kvs(input: TokenStream) -> TokenStream {
    let input = input.to_string();

    let mut output = String::new();
    let mut ty = None;
    for s in input.split_whitespace() {
        match ty.take() {
            Some(ty) => {
                output.push_str(&Name::analyze(s).build(ty));
                output.push('\n');
            }
            None => {
                ty = Some(s);
            }
        }
    }
    output.parse().unwrap()
}

struct Name<'a> {
    format: &'a str,
    upper_ident: String,
    lower_ident: String,
    arg_list: String,
}

impl<'a> Name<'a> {
    /// 分析名字，确保大括号闭合、不嵌套且不跨越标识符
    fn analyze(name: &'a str) -> Self {
        // 去除引号
        let name = name
            .trim()
            .strip_prefix('"')
            .unwrap()
            .strip_suffix('"')
            .unwrap();

        let mut args = HashSet::new();
        let mut brace = usize::MAX;

        let mut ident_collector = IdentCollector::new();
        let mut arg_list = String::new();

        for (i, c) in name.char_indices() {
            match c {
                '.' => {
                    assert_eq!(brace, usize::MAX);
                    ident_collector.push_();
                }
                '{' => {
                    assert_eq!(brace, usize::MAX);
                    brace = i;
                    ident_collector.push_();
                }
                '}' => {
                    assert_ne!(brace, usize::MAX);
                    let arg = &name[brace + 1..i];
                    brace = usize::MAX;
                    ident_collector.push_();

                    if !args.insert(arg) {
                        panic!("Duplicate argument {arg} in name {name}");
                    }
                    if !arg_list.is_empty() {
                        arg_list.push_str(", ");
                    }
                    arg_list.push_str(arg);
                    arg_list.push_str(": &str");
                }
                _ => {
                    assert!(
                        c.is_ascii_alphanumeric() || c == '_',
                        "Invalid character '{c}' in name \"{name}\""
                    );
                    ident_collector.pushc(c);
                }
            }
        }

        let (upper_ident, lower_ident) = ident_collector.build();
        Self {
            format: name,
            upper_ident,
            lower_ident,
            arg_list,
        }
    }

    fn build(&self, ty: &str) -> String {
        let ty = ty.trim();
        let key = self.to_key();
        let read_fn = self.to_read_fn(ty);
        let write_fn = self.to_write_fn(ty);

        format!(
            "\
{key}
{read_fn}
{write_fn}
"
        )
    }

    fn to_key(&self) -> String {
        let Self {
            format,
            upper_ident,
            lower_ident,
            arg_list,
        } = self;

        if arg_list.is_empty() {
            format!("pub const {upper_ident}: &str = \"{format}\";")
        } else {
            format!(
                "\
pub fn {lower_ident}({arg_list}) -> String {{
    format!(\"{format}\")
}}"
            )
        }
    }

    fn to_read_fn(&self, ty: &str) -> String {
        #[rustfmt::skip]
        fn read_fn(name: &str, ty: &str, f: &str) -> String {
format!(
"impl<'a> crate::GGufReader<'a> {{
    pub fn read_{name}_val(&mut self) -> Result<{ty}, crate::GGufReadError> {{
        self.{f}()
    }}
}}")
        }

        let name = &self.lower_ident;
        match ty {
            #[rustfmt::skip]
            "u8"  |
            "i8"  |
            "u16" |
            "i16" |
            "u32" |
            "i32" |
            "f32" |
            "u64" |
            "i64" |
            "f64" => read_fn(name, ty, "read"),
            "bool" => read_fn(name, ty, "read_bool"),
            "str" => read_fn(name, "&'a str", "read_str"),
            #[rustfmt::skip]
            "[u8]"  |
            "[i8]"  |
            "[u16]" |
            "[i16]" |
            "[u32]" |
            "[i32]" |
            "[f32]" |
            "[u64]" |
            "[i64]" |
            "[f64]" |
            "[bool]"|
            "[str]" => {
                // TODO
                String::new()
            }
            _ => todo!("Unsupported type {ty}"),
        }
    }

    fn to_write_fn(&self, ty: &str) -> String {
        #[rustfmt::skip]
        fn write_fn(name: &str, ty: &str, f: &str) -> String {
format!(
"impl<T: std::io::Write> crate::GGufWriter<T> {{
    pub fn write_{name}_val(&mut self, val: {ty}) -> std::io::Result<()> {{
        {f}
    }}
}}")
        }

        let name = &self.lower_ident;
        match ty {
            #[rustfmt::skip]
            "u8"  |
            "i8"  |
            "u16" |
            "i16" |
            "u32" |
            "i32" |
            "f32" |
            "u64" |
            "i64" |
            "f64" => write_fn(name, ty, "self.write(&[val])"),
            "bool" => write_fn(name, ty, "self.write(if val { &[1u8] } else { &[0u8] })"),
            "str" => write_fn(name, "&str", "self.write_str(val)"),
            #[rustfmt::skip]
            "[u8]"  |
            "[i8]"  |
            "[u16]" |
            "[i16]" |
            "[u32]" |
            "[i32]" |
            "[f32]" |
            "[u64]" |
            "[i64]" |
            "[f64]" => write_fn(name, &format!("&{ty}"), "self.write(val)"),
            "[bool]" => write_fn(
                name,
                "&[bool]",
                "for b in val { self.write(if b { &[1u8] } else { &[0u8] })? } Ok(())",
            ),
            "[str]" => write_fn(
                name,
                "&[&str]",
                "for s in val { self.write_str(s)? } Ok(())",
            ),
            _ => todo!("Unsupported type {ty}"),
        }
    }
}
