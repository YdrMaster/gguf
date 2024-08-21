use std::str::FromStr;

#[derive(Clone, PartialEq, Debug)]
#[repr(u8)]
pub enum Type {
    Default,
    LoRA,
    Vocab,
}

impl FromStr for Type {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "LoRA" => Ok(Self::LoRA),
            "vocab" => Ok(Self::Vocab),
            _ => Err(()),
        }
    }
}
