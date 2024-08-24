pub(crate) struct IdentCollector {
    upper_ident: String,
    lower_ident: String,
    distinct: bool,
}

impl IdentCollector {
    pub fn new() -> Self {
        Self {
            upper_ident: String::new(),
            lower_ident: String::new(),
            distinct: true,
        }
    }

    pub fn push_(&mut self) {
        if !self.distinct {
            self.upper_ident.push('_');
            self.lower_ident.push('_');
            self.distinct = true;
        }
    }

    pub fn pushc(&mut self, c: char) {
        self.upper_ident.push(c.to_ascii_uppercase());
        self.lower_ident.push(c.to_ascii_lowercase());
        self.distinct = false;
    }

    pub fn build(self) -> (String, String) {
        (
            self.upper_ident.trim_end_matches('_').into(),
            self.lower_ident.trim_end_matches('_').into(),
        )
    }
}
